"""Tests for the BAxUS generator."""

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from botorch.acquisition import LogExpectedImprovement
from pydantic import ValidationError
from xopt import VOCS, Xopt
from xopt import Evaluator as XoptEvaluator
from xopt.errors import GeneratorWarning, VOCSError
from xopt.generator import StateOwner
from xopt.generators import get_generator_dynamic, list_available_generators
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.expected_improvement import (
    ExpectedImprovementGenerator,
)
from xopt.generators.bayesian.baxus import (
    BAxUSEmbedding,
    BAxUSGenerator,
    BAxUSTrustRegion,
    _normalize_lengthscale_weights,
)
from xopt.generators.bayesian.objectives import CustomXoptObjective
from xopt.vocs import ContextualVariable, DiscreteVariable


@pytest.fixture(autouse=True)
def _quiet_missing_eval_budget():
    """Silence the eval_budget advisory across this module.

    Most tests here construct generators without ``eval_budget`` on purpose,
    because they exercise something unrelated to the expansion schedule, and the
    advisory would otherwise fire on nearly every construction. The contract
    itself stays pinned by ``TestBAxUSEvalBudgetWarning``, whose ``pytest.warns``
    and ``catch_warnings`` blocks install their own filters over this one.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="BAxUS: eval_budget is not set",
            category=GeneratorWarning,
        )
        yield


def _simple_vocs(n_vars: int = 6) -> VOCS:
    """Create a VOCS with *n_vars* variables for testing."""
    variables = {f"x{i}": [-1.0, 1.0] for i in range(n_vars)}
    return VOCS(variables=variables, objectives={"f": "MAXIMIZE"})


def _sphere(inputs: dict[str, float]) -> dict[str, float]:
    """Negative sphere - maximum at the origin."""
    return {"f": -sum(v**2 for v in inputs.values())}


def _assert_point_in_bounds(point: dict[str, float], vocs: VOCS) -> None:
    """Assert every variable value lies within its VOCS domain."""
    for name in vocs.variable_names:
        lo, hi = vocs.variables[name].domain
        assert lo <= point[name] <= hi, f"{name}={point[name]} outside [{lo}, {hi}]"


class _MinimalCustomObjective(CustomXoptObjective):
    """Concrete CustomXoptObjective, just enough to exercise the rejection path
    (CustomXoptObjective is abstract, so a bare instance can't reach our validator)."""

    def forward(
        self, samples: torch.Tensor, X: torch.Tensor | None = None
    ) -> torch.Tensor:
        return samples


def _seed_sobol(gen: BAxUSGenerator) -> None:
    """Run the Sobol init phase so subsequent generate() calls take the BO path."""
    for _ in range(gen.n_initial_sobol):
        point = gen.generate(1)[0]
        gen.add_data(pd.DataFrame([{**point, **_sphere(point)}]))


def _bo_step(gen: BAxUSGenerator) -> dict[str, float]:
    """Run one BO generate->evaluate->add_data cycle, asserting the candidate stays in bounds."""
    point = gen.generate(1)[0]
    _assert_point_in_bounds(point, gen.vocs)
    gen.add_data(pd.DataFrame([{**point, **_sphere(point)}]))
    return point


def _force_expansion(gen: BAxUSGenerator) -> None:
    """Put the trust region below length_min with an unbeatable incumbent, so the
    next fold-in trips the restart -> expand branch."""
    gen.trust_region.length = gen.trust_region.length_min / 2.0
    gen.trust_region.best_value = 1e9  # guarantee a failure (no improvement)


def _round_trip(gen: BAxUSGenerator, *, with_data: bool = False) -> BAxUSGenerator:
    """Dump -> construct, the documented resume path.

    The live botorch model is not serializable and is popped by hand, exactly as
    the docs instruct a user to do.
    """
    dump = gen.model_dump()
    dump.pop("model", None)
    restored = BAxUSGenerator(**dump)
    if with_data and gen.data is not None:
        restored.data = gen.data.copy()
    return restored


def _fold_in_results(gen: BAxUSGenerator) -> None:
    """Advance the trust region over results ingested since the last generate.

    The trust region is advanced at the start of ``generate`` (mirroring how
    ``BayesianGenerator`` drives ``TurboController.update_state``), so folding in
    the most recent evaluations means asking for the next candidate.
    """
    gen.generate(1)


def _set_target_dim(gen: BAxUSGenerator, target_dim: int) -> None:
    """Move the generator to a given target dimensionality, keeping the
    embedding and trust region consistent (the generator enforces that)."""
    gen.embedding = BAxUSEmbedding.create(
        input_dim=gen.embedding.input_dim, target_dim=target_dim, seed=0
    )
    gen.trust_region = BAxUSTrustRegion(target_dim=target_dim)


class TestBAxUSGeneratorCreation:
    """Test generator initialisation."""

    def test_baxus_generator_creation(self) -> None:
        gen = BAxUSGenerator(vocs=_simple_vocs())

        assert gen.name == "baxus"
        assert gen.trust_region.target_dim == 2
        assert gen.embedding.target_dim == 2
        assert gen.embedding.input_dim == 6

    def test_custom_target_dim(self) -> None:
        gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=10), target_dim_init=4)
        assert gen.trust_region.target_dim == 4
        assert gen.embedding.target_dim == 4
        assert gen.embedding.input_dim == 10

    def test_target_dim_capped_to_input_dim(self) -> None:
        gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=3), target_dim_init=10)
        assert gen.trust_region.target_dim == 3

    def test_missing_vocs_raises_validation_error(self) -> None:
        """Without vocs, ``_init_components`` must skip auto-creation and let
        pydantic report the missing required fields, not crash internally."""
        with pytest.raises(ValidationError, match="vocs"):
            BAxUSGenerator()

    def test_embedding_structure(self) -> None:
        """Each column of S should have exactly one non-zero entry."""
        gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=8), target_dim_init=3)
        S = np.asarray(gen.embedding.matrix)
        for col_idx in range(S.shape[1]):
            assert np.count_nonzero(S[:, col_idx]) == 1
            assert abs(S[:, col_idx]).max() == 1.0


class TestBAxUSInitialSobol:
    """Test that early generate() calls return Sobol points in bounds."""

    def test_baxus_initial_sobol(self) -> None:
        vocs = _simple_vocs()
        gen = BAxUSGenerator(vocs=vocs)

        for _ in range(gen.n_initial_sobol):
            candidates = gen.generate(1)
            assert len(candidates) == 1
            _assert_point_in_bounds(candidates[0], vocs)
            gen.add_data(pd.DataFrame([{**candidates[0], **_sphere(candidates[0])}]))


class TestBAxUSSeed:
    """The ``seed`` field pins the embedding and Sobol initialisation.

    GP fitting/acquisition use torch's global RNG and are out of scope.
    """

    def test_same_seed_gives_identical_embedding(self) -> None:
        vocs = _simple_vocs(n_vars=6)
        g1 = BAxUSGenerator(vocs=vocs, seed=0)
        g2 = BAxUSGenerator(vocs=vocs, seed=0)
        assert g1.embedding.matrix == g2.embedding.matrix

    def test_different_seeds_give_different_embedding(self) -> None:
        vocs = _simple_vocs(n_vars=6)
        g1 = BAxUSGenerator(vocs=vocs, seed=0)
        g2 = BAxUSGenerator(vocs=vocs, seed=1)
        assert g1.embedding.matrix != g2.embedding.matrix

    def test_seed_none_still_works(self) -> None:
        vocs = _simple_vocs(n_vars=4)
        gen = BAxUSGenerator(vocs=vocs, seed=None)
        point = gen.generate(1)[0]
        _assert_point_in_bounds(point, vocs)


class TestBAxUSSobolSequence:
    """The Sobol phase must be one reproducible low-discrepancy sequence,
    not independent scrambles from a fresh engine per call."""

    def test_same_seed_gives_identical_sobol_points(self) -> None:
        vocs = _simple_vocs(n_vars=6)
        g1 = BAxUSGenerator(vocs=vocs, seed=7)
        g2 = BAxUSGenerator(vocs=vocs, seed=7)
        pts1 = [g1.generate(1)[0] for _ in range(g1.n_initial_sobol)]
        pts2 = [g2.generate(1)[0] for _ in range(g2.n_initial_sobol)]
        assert pts1 == pts2

    def test_sobol_points_advance(self) -> None:
        """A fresh ``draw(1)`` engine per call would repeat the same first point."""
        gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=6), seed=3)
        pts = [tuple(gen.generate(1)[0].values()) for _ in range(gen.n_initial_sobol)]
        assert len(set(pts)) == len(pts)


class TestBAxUSEndToEnd:
    """End-to-end: Sobol init then BO steps."""

    def test_baxus_end_to_end(self) -> None:
        gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=4), target_dim_init=2)

        _seed_sobol(gen)
        for _ in range(3):
            _bo_step(gen)

        assert gen.data is not None
        assert len(gen.data) == gen.n_initial_sobol + 3


def _asymmetric_vocs() -> VOCS:
    """VOCS with per-variable bounds at different scales - catches bounds orientation bugs."""
    return VOCS(
        variables={"x0": [0.0, 10.0], "x1": [-165.0, 165.0], "x2": [0.5, 0.6]},
        objectives={"f": "MAXIMIZE"},
    )


class TestBAxUSAsymmetricBounds:
    """Guard against the v2->v3 `vocs.bounds` orientation flip (was (2, D), now (D, 2))."""

    def test_points_lie_in_per_variable_bounds(self) -> None:
        vocs = _asymmetric_vocs()
        gen = BAxUSGenerator(vocs=vocs, target_dim_init=2)

        _seed_sobol(gen)
        # A wrong bounds orientation produces out-of-box points in the BO phase.
        for _ in range(3):
            _bo_step(gen)


class TestBAxUSSteering:
    """BO-phase points must be better *on average* than the Sobol seed points.

    Guards against a min-form/max-form inversion (``get_objective_data``
    negates MAXIMIZE objectives). Never compare best-over-all-data against
    best-over-seeds - the seeds are a subset of all data, so that is
    vacuously true and lets the generator anti-optimize undetected.

    The optimum is deliberately placed OFF-CENTER. With an optimum at the origin
    - the center of the symmetric [-1, 1] domain - these tests only measure
    "did the candidate move toward the middle of the box", which a generator
    that ignores the acquisition function entirely (proposing a constant zero in
    embedded space) satisfies perfectly. The offset makes them measure
    optimization instead.
    """

    N_SOBOL = 6
    N_BO_STEPS = 15
    OPTIMUM = 0.6

    def _run_phases(self, direction: str) -> tuple[list[float], list[float]]:
        """Run Sobol seeding then BO steps; return (sobol_ys, bo_ys).

        The objective peaks at ``x_i = OPTIMUM`` for every i: MAXIMIZE gets
        -||x - c||^2 (values <= 0), MINIMIZE gets +||x - c||^2 (values >= 0).
        """
        torch.manual_seed(0)
        vocs = VOCS(
            variables={f"x{i}": [-1.0, 1.0] for i in range(6)},
            objectives={"f": direction},
        )
        gen = BAxUSGenerator(
            vocs=vocs,
            target_dim_init=2,
            n_initial_sobol=self.N_SOBOL,
            seed=0,
            eval_budget=self.N_SOBOL + self.N_BO_STEPS,
        )
        sign = -1.0 if direction == "MAXIMIZE" else 1.0

        ys: list[float] = []
        for _ in range(self.N_SOBOL + self.N_BO_STEPS):
            point = gen.generate(1)[0]
            _assert_point_in_bounds(point, vocs)
            y = sign * sum((v - self.OPTIMUM) ** 2 for v in point.values())
            ys.append(y)
            gen.add_data(pd.DataFrame([{**point, "f": y}]))
        return ys[: self.N_SOBOL], ys[self.N_SOBOL :]

    def test_maximize_bo_phase_beats_sobol_phase(self) -> None:
        sobol_ys, bo_ys = self._run_phases("MAXIMIZE")
        assert np.mean(bo_ys) > np.mean(sobol_ys), (
            f"MAXIMIZE BO phase is worse than random seeding (direction inverted?): "
            f"sobol mean={np.mean(sobol_ys):.4f} bo mean={np.mean(bo_ys):.4f}"
        )

    def test_minimize_bo_phase_beats_sobol_phase(self) -> None:
        sobol_ys, bo_ys = self._run_phases("MINIMIZE")
        assert np.mean(bo_ys) < np.mean(sobol_ys), (
            f"MINIMIZE BO phase is worse than random seeding (direction inverted?): "
            f"sobol mean={np.mean(sobol_ys):.4f} bo mean={np.mean(bo_ys):.4f}"
        )


class TestBAxUSNaNIngestion:
    """NaN objectives (failed evaluations) must not poison the training data
    or the trust-region update."""

    def test_nan_row_dropped_from_training_data(self) -> None:
        gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=4), target_dim_init=2)

        point = gen.generate(1)[0]
        gen.add_data(pd.DataFrame([{**point, "f": float("nan")}]))

        assert len(gen.data) == 1  # row is kept in the canonical history...
        assert gen._finite_data().empty  # ...but excluded from GP training

    def test_mixed_batch_keeps_only_finite_rows(self) -> None:
        gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=4), target_dim_init=2)

        rows = []
        for y in (1.0, float("nan"), 2.0):
            point = gen.generate(1)[0]
            rows.append({**point, "f": y})
        gen.add_data(pd.DataFrame(rows))

        assert len(gen.data) == 3
        assert len(gen._finite_data()) == 2

    def test_bo_step_survives_nan_ingestion(self) -> None:
        """After a NaN row mid-run, the next BO candidate is still finite and in bounds."""
        vocs = _simple_vocs(n_vars=4)
        gen = BAxUSGenerator(vocs=vocs, target_dim_init=2)

        _seed_sobol(gen)
        point = gen.generate(1)[0]
        gen.add_data(pd.DataFrame([{**point, "f": float("nan")}]))

        candidate = gen.generate(1)[0]
        assert all(math.isfinite(v) for v in candidate.values())
        _assert_point_in_bounds(candidate, vocs)


class TestBAxUSTrustRegionSeedPhase:
    """The trust region must track BO-phase evaluations only: the BoTorch
    reference starts BO with a pristine trust region, so Sobol seeding must
    not move the counters or ``best_value``."""

    def test_seed_phase_leaves_state_pristine(self) -> None:
        gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=4), target_dim_init=2, seed=0)

        _seed_sobol(gen)

        assert gen.trust_region.length == gen.trust_region.length_init
        assert gen.trust_region.success_counter == 0
        assert gen.trust_region.failure_counter == 0
        assert gen.trust_region.best_value is None

    def test_first_bo_result_updates_state(self) -> None:
        gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=4), target_dim_init=2, seed=0)

        _seed_sobol(gen)
        _bo_step(gen)
        assert gen.trust_region.best_value is None  # not folded in yet
        _fold_in_results(gen)

        assert gen.trust_region.best_value is not None
        assert gen.tr_observed_rows == len(gen.data)

    def test_straddling_batch_counts_only_post_quota_rows(self) -> None:
        """A batch that crosses the seed quota updates the trust region with the
        BO-phase rows only."""
        vocs = _simple_vocs(n_vars=4)
        gen = BAxUSGenerator(vocs=vocs, target_dim_init=2, n_initial_sobol=2, seed=0)

        pts = [gen.generate(1)[0] for _ in range(3)]
        rows = [
            {**pts[0], "f": 100.0},
            {**pts[1], "f": 100.0},
            {**pts[2], "f": 1.0},
        ]
        gen.add_data(pd.DataFrame(rows))
        _fold_in_results(gen)

        # MAXIMIZE ⇒ weighted value is +f; only the BO row (f=1.0) should count.
        assert gen.trust_region.best_value == 1.0

    def test_trust_region_is_independent_of_ingestion_chunking(self) -> None:
        """Same observations, different add_data batching ⇒ identical state.

        The state machine is driven by generate calls, so feeding results one at
        a time or all at once must not change the trajectory (a warm start via
        ``Xopt(..., data=...)`` arrives as one big frame).
        """
        vocs = _simple_vocs(n_vars=4)
        rng = np.random.default_rng(0)
        pts = [
            {f"x{i}": float(v) for i, v in enumerate(rng.uniform(-1, 1, 4))}
            for _ in range(12)
        ]
        rows = [{**p, **_sphere(p)} for p in pts]

        def state(batched: bool) -> tuple[float, int, int, float]:
            gen = BAxUSGenerator(
                vocs=vocs, target_dim_init=2, n_initial_sobol=2, seed=0
            )
            if batched:
                gen.add_data(pd.DataFrame(rows))
            else:
                for row in rows:
                    gen.add_data(pd.DataFrame([row]))
            _fold_in_results(gen)
            tr = gen.trust_region
            return (tr.length, tr.success_counter, tr.failure_counter, tr.best_value)

        assert state(batched=True) == state(batched=False)

    def test_baxus_embedding_expansion(self) -> None:
        """Expansion grows target_dim, resets the region, and preserves best_value."""
        gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=8), target_dim_init=2, seed=0)
        _seed_sobol(gen)

        # Force the next fold-in to trip the restart branch.
        _force_expansion(gen)
        _bo_step(gen)
        _fold_in_results(gen)

        assert gen.trust_region.target_dim > 2
        assert gen.embedding.target_dim == gen.trust_region.target_dim
        assert gen.trust_region.length == gen.trust_region.length_init
        assert gen.trust_region.best_value == 1e9
        assert not gen.trust_region.restart_triggered
        assert gen.n_expansions == 1


class TestBAxUSBatchGeneration:
    """``generate(2)`` must raise (matching xopt's guard), not silently
    under-deliver one point."""

    def test_generate_multiple_candidates_raises(self) -> None:
        gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=4), target_dim_init=2)
        with pytest.raises(NotImplementedError, match="parallel candidate generation"):
            gen.generate(2)


class TestBAxUSTrainModel:
    """train_model must refuse to fit a GP with no finite-objective data."""

    def test_raises_when_no_data_available(self) -> None:
        gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=4), target_dim_init=2)
        with pytest.raises(ValueError, match="no data available to build model"):
            gen.train_model()


class TestBAxUSViaXopt:
    """Full integration through the Xopt runner."""

    def test_baxus_via_xopt(self) -> None:
        vocs = _simple_vocs(n_vars=4)
        gen = BAxUSGenerator(vocs=vocs, target_dim_init=2)
        evaluator = XoptEvaluator(function=_sphere)
        X = Xopt(evaluator=evaluator, generator=gen)

        # Seed with random points (like the real phases.py flow), then BO.
        X.random_evaluate(gen.n_initial_sobol)
        for _ in range(3):
            X.step()

        assert len(X.data) == gen.n_initial_sobol + 3

        for name in vocs.variable_names:
            lo, hi = vocs.variables[name].domain
            values = X.data[name].to_numpy()
            assert np.all(values >= lo - 1e-9), f"{name} below lower bound"
            assert np.all(values <= hi + 1e-9), f"{name} above upper bound"


class TestBAxUSLengthscaleWeights:
    """_lengthscale_weights: gpytorch kernel -> duck-typed hook -> uniform fallback."""

    def _gen(self, target_dim: int = 2) -> BAxUSGenerator:
        return BAxUSGenerator(vocs=_simple_vocs(n_vars=4), target_dim_init=target_dim)

    @staticmethod
    def _kernel_model(lengthscale: torch.Tensor) -> MagicMock:
        """A mock shaped like a gpytorch model: covar_module.base_kernel.lengthscale."""
        kernel = MagicMock()
        kernel.lengthscale = lengthscale
        covar = MagicMock()
        covar.base_kernel = kernel
        model = MagicMock(spec=["covar_module"])
        model.covar_module = covar
        return model

    def test_gpytorch_kernel_lengthscale_used(self) -> None:
        gen = self._gen()
        model = self._kernel_model(torch.tensor([[0.5, 2.0]], dtype=torch.double))

        weights = gen._lengthscale_weights(model)
        assert weights.shape == (2,)
        assert not torch.allclose(weights, torch.ones(2, dtype=torch.double))

    def test_scalar_lengthscale_unsqueezed(self) -> None:
        gen = self._gen(target_dim=1)
        model = self._kernel_model(torch.tensor(0.5, dtype=torch.double))

        assert gen._lengthscale_weights(model).shape == (1,)

    def test_duck_typed_hook_used_when_no_kernel(self) -> None:
        gen = self._gen()
        model = MagicMock(spec=["get_lengthscale_weights"])
        model.get_lengthscale_weights.return_value = torch.tensor(
            [0.5, 1.5], dtype=torch.double
        )

        weights = gen._lengthscale_weights(model)
        assert torch.allclose(weights, torch.tensor([0.5, 1.5], dtype=torch.double))
        model.get_lengthscale_weights.assert_called_once_with(2)

    def test_uniform_fallback_when_hook_returns_none(self) -> None:
        gen = self._gen()
        model = MagicMock(spec=["get_lengthscale_weights"])
        model.get_lengthscale_weights.return_value = None

        assert torch.allclose(
            gen._lengthscale_weights(model), torch.ones(2, dtype=torch.double)
        )

    def test_uniform_fallback_for_bare_model(self) -> None:
        gen = self._gen()
        model = MagicMock(spec=["posterior", "num_outputs"])

        assert torch.allclose(
            gen._lengthscale_weights(model), torch.ones(2, dtype=torch.double)
        )


class TestBAxUSConfig:
    def test_explicit_n_initial_sobol_kept(self) -> None:
        gen = BAxUSGenerator(
            vocs=_simple_vocs(n_vars=4), target_dim_init=2, n_initial_sobol=5
        )
        assert gen.n_initial_sobol == 5

    def test_auto_n_initial_sobol(self) -> None:
        gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=6), target_dim_init=3)
        assert gen.n_initial_sobol == 4  # max(2, target_dim + 1)


class TestBAxUSTrustRegion:
    """Trust-region state bookkeeping (weighted convention: higher is better)."""

    def test_success_expands_length(self) -> None:
        tr = BAxUSTrustRegion(target_dim=2, length=0.8)
        tr.best_value = 0.0
        for i in range(tr.success_tolerance):
            tr.update(torch.tensor([10.0 * (i + 1)]))
        assert tr.length > 0.8

    def test_failure_shrinks_length(self) -> None:
        tr = BAxUSTrustRegion(target_dim=2, length=0.8)
        tr.best_value = 999.0
        for _ in range(tr.failure_tolerance):
            tr.update(torch.tensor([-999.0]))
        assert tr.length < 0.8

    def test_restart_triggered_on_tiny_length(self) -> None:
        tr = BAxUSTrustRegion(target_dim=2, length=0.008)
        tr.best_value = 999.0
        for _ in range(100):
            tr.update(torch.tensor([-999.0]))
            if tr.restart_triggered:
                break
        assert tr.restart_triggered

    def test_first_update_counts_as_success(self) -> None:
        """best_value=None (pristine) means the first observed batch always improves."""
        tr = BAxUSTrustRegion(target_dim=2)
        tr.update(torch.tensor([-5.0]))
        assert tr.best_value == -5.0
        assert tr.failure_counter == 0

    def test_failure_tolerance_scales_with_target_dim(self) -> None:
        assert BAxUSTrustRegion(target_dim=2).failure_tolerance == 1
        assert BAxUSTrustRegion(target_dim=10).failure_tolerance == 5

    def test_round_trips_through_dump(self) -> None:
        tr = BAxUSTrustRegion(
            target_dim=3, length=0.4, success_counter=2, best_value=1.5
        )
        assert BAxUSTrustRegion(**tr.model_dump()) == tr


@dataclass
class _TutorialBaxusState:
    """Oracle: the reference ``BaxusState`` from the BoTorch BAxUS tutorial,
    trimmed to the fields the failure-tolerance formula needs."""

    dim: int
    eval_budget: int
    new_bins_on_split: int = 3
    d_init: int = 0
    target_dim: int = 0
    n_splits: int = 0
    length_init: float = 0.8
    length_min: float = 0.5**7

    def __post_init__(self) -> None:
        n_splits = round(math.log(self.dim, self.new_bins_on_split + 1))
        self.d_init = int(
            1
            + np.argmin(
                np.abs(
                    (1 + np.arange(self.new_bins_on_split))
                    * (1 + self.new_bins_on_split) ** n_splits
                    - self.dim
                )
            )
        )
        self.target_dim = self.d_init
        self.n_splits = n_splits

    @property
    def split_budget(self) -> int:
        growth: int = (self.new_bins_on_split + 1) ** (self.n_splits + 1)
        return round(
            -1
            * (self.new_bins_on_split * self.eval_budget * self.target_dim)
            / (self.d_init * (1 - growth))
        )

    @property
    def failure_tolerance(self) -> int:
        if self.target_dim == self.dim:
            return self.target_dim
        k = math.floor(math.log(self.length_min / self.length_init, 0.5))
        return min(self.target_dim, max(1, math.floor(self.split_budget / k)))


class TestBAxUSFailureTolerance:
    """Budget-aware failure tolerance (paper Alg. 1) against the tutorial oracle.

    input_dim=16 with b=3 is an exact power of b+1, where the oracle's derived
    d_init (=1) and split count (=2) coincide with ours - the regime in which a
    verbatim comparison is meaningful.
    """

    def _gen(self, eval_budget: int | None, target_dim_init: int = 1) -> BAxUSGenerator:
        return BAxUSGenerator(
            vocs=_simple_vocs(n_vars=16),
            target_dim_init=target_dim_init,
            n_initial_sobol=10,
            seed=0,
            eval_budget=eval_budget,
        )

    # bo_budget=93 with target_dim=8 is the discriminating pair: computing the
    # budget without subtracting the 10 seed points gives 6 instead of 5, so
    # this pins the total-including-seeds semantics of eval_budget.
    @pytest.mark.parametrize("bo_budget", [30, 93, 200])
    def test_matches_tutorial_oracle_across_dims(self, bo_budget: int) -> None:
        oracle = _TutorialBaxusState(dim=16, eval_budget=bo_budget)
        assert oracle.d_init == 1  # sanity: oracle derives the d_init we configure
        gen = self._gen(
            eval_budget=bo_budget + 10
        )  # our field is total incl. the 10 seeds

        for target_dim in (1, 4, 8, 16):
            oracle.target_dim = target_dim
            _set_target_dim(gen, target_dim)
            gen._refresh_failure_tolerance()
            assert gen.trust_region.failure_tolerance == oracle.failure_tolerance

    def test_differs_from_heuristic(self) -> None:
        """The paper schedule is not just ceil(d/2) in disguise."""
        gen = self._gen(eval_budget=210)
        _set_target_dim(gen, 4)
        gen._refresh_failure_tolerance()
        assert gen.trust_region.failure_tolerance == 4  # heuristic would say 2

    def test_heuristic_kept_without_budget(self) -> None:
        gen = self._gen(eval_budget=None, target_dim_init=4)
        assert gen.trust_region.failure_tolerance == 2  # ceil(4/2)
        gen._refresh_failure_tolerance()
        assert gen.trust_region.failure_tolerance == 2

    def test_full_dim_widens_to_target_dim(self) -> None:
        """At full dimensionality the tolerance is target_dim, budget or not."""
        gen = self._gen(eval_budget=None, target_dim_init=16)
        gen._refresh_failure_tolerance()
        assert gen.trust_region.failure_tolerance == 16

    def test_bo_phase_applies_budget_tolerance(self) -> None:
        """Folding in a BO result refreshes the tolerance (also covers a d_init the
        reference would not derive itself: n_splits generalizes)."""
        vocs = _simple_vocs(n_vars=16)
        gen = BAxUSGenerator(
            vocs=vocs, target_dim_init=4, n_initial_sobol=2, seed=0, eval_budget=210
        )
        _seed_sobol(gen)
        _bo_step(gen)
        _fold_in_results(gen)
        assert (
            gen.trust_region.failure_tolerance == 4
        )  # paper value; heuristic would say 2

    def test_expansion_recomputes_tolerance(self) -> None:
        vocs = _simple_vocs(n_vars=8)
        gen = BAxUSGenerator(vocs=vocs, target_dim_init=2, seed=0, eval_budget=50)
        _seed_sobol(gen)

        _force_expansion(gen)
        _bo_step(gen)
        _fold_in_results(gen)

        assert gen.trust_region.target_dim == 8  # expanded to full dim
        assert gen.trust_region.failure_tolerance == 8

    def test_expansion_recomputes_budget_tolerance_below_full_dim(self) -> None:
        """An expansion landing below input_dim exercises the budget formula,
        not just the full-dim widening."""
        vocs = _simple_vocs(n_vars=32)
        gen = BAxUSGenerator(vocs=vocs, target_dim_init=2, seed=0, eval_budget=203)
        _seed_sobol(gen)

        _force_expansion(gen)
        _bo_step(gen)
        _fold_in_results(gen)

        assert gen.trust_region.target_dim == 8  # 2 * (b + 1), still < 32
        # d_init=2, n_splits=2, budget=200: round(4800/126)=38 -> floor(38/6)=6
        assert gen.trust_region.failure_tolerance == 6

    def test_explicit_failure_tolerance_survives_construction(self) -> None:
        """Static-model semantics only: failure_tolerance is derived state and
        the generator re-derives it on every BO-phase ingest."""
        assert (
            BAxUSTrustRegion(target_dim=8, failure_tolerance=7).failure_tolerance == 7
        )

    def test_budget_below_seed_quota_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must cover the seed quota"):
            self._gen(eval_budget=5)  # quota is 10

    def test_dump_round_trips_budget_and_tolerance(self) -> None:
        gen = self._gen(eval_budget=210)
        _set_target_dim(gen, 4)
        gen._refresh_failure_tolerance()

        restored = _round_trip(gen)

        assert restored.eval_budget == 210
        assert (
            restored.trust_region.failure_tolerance
            == gen.trust_region.failure_tolerance
        )


class TestBAxUSPostExpansionFit:
    """Right after an embedding expansion the projected training data has
    duplicated coordinates, which can make the covariance indefinite; the run
    must survive the degraded fit and log it."""

    def test_generate_survives_post_expansion_training(self) -> None:
        vocs = _simple_vocs(n_vars=8)
        gen = BAxUSGenerator(vocs=vocs, target_dim_init=2, seed=0)
        _seed_sobol(gen)

        _force_expansion(gen)
        _bo_step(gen)

        # this generate expands first, then trains and proposes in the new space
        candidate = gen.generate(1)[0]
        assert gen.embedding.target_dim == 8
        _assert_point_in_bounds(candidate, vocs)
        assert all(math.isfinite(v) for v in candidate.values())

    def test_fit_failure_warning_surfaces_in_logs(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """train_model must mirror xopt's fit-failure warning into the logger
        and still re-emit the original warning unchanged."""
        gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=4), target_dim_init=2, seed=0)
        _seed_sobol(gen)

        constructor_cls = type(gen.gp_constructor)
        real_build = constructor_cls.build_model

        def failing_build(self: Any, **kwargs: Any) -> Any:
            warnings.warn(
                "Model fitting failed. Returning untrained model.", stacklevel=1
            )
            return real_build(self, **kwargs)

        monkeypatch.setattr(constructor_cls, "build_model", failing_build)

        logger_name = "xopt.generators.bayesian.baxus"
        with (
            caplog.at_level(logging.WARNING, logger=logger_name),
            pytest.warns(UserWarning, match="Model fitting failed"),
        ):
            gen.train_model()

        assert any(
            "GP fit failed at target_dim=2" in record.message
            for record in caplog.records
        )


class TestBAxUSEmbedding:
    """Sparse random embedding: structure, projection round-trip, expansion."""

    def test_create_structure(self) -> None:
        """Each input column has exactly one signed unit entry."""
        emb = BAxUSEmbedding.create(input_dim=8, target_dim=3, seed=0)
        S = np.asarray(emb.matrix)
        assert S.shape == (3, 8)
        for col in range(8):
            assert np.count_nonzero(S[:, col]) == 1
            assert abs(S[:, col]).max() == 1.0

    def test_create_assigns_both_signs(self) -> None:
        """Sign randomization is what makes the embedding an unbiased projection;
        an all-positive matrix is still structurally valid, so check the signs."""
        emb = BAxUSEmbedding.create(input_dim=200, target_dim=4, seed=0)
        matrix = np.asarray(emb.matrix)

        positive = int((matrix == 1.0).sum())
        negative = int((matrix == -1.0).sum())
        assert positive and negative, "embedding signs are not randomized"
        assert 0.3 < positive / (positive + negative) < 0.7

    def test_create_is_seed_deterministic(self) -> None:
        e1 = BAxUSEmbedding.create(input_dim=6, target_dim=2, seed=0)
        e2 = BAxUSEmbedding.create(input_dim=6, target_dim=2, seed=0)
        e3 = BAxUSEmbedding.create(input_dim=6, target_dim=2, seed=1)
        assert e1.matrix == e2.matrix
        assert e1.matrix != e3.matrix

    def test_lift_project_round_trip(self) -> None:
        """Points lifted from target space project back exactly (S rows are orthogonal)."""
        emb = BAxUSEmbedding.create(input_dim=6, target_dim=2, seed=0)
        Z = np.array([[0.5, -0.25], [1.0, 1.0]])
        assert np.allclose(emb.project(emb.lift(Z)), Z)

    def test_expand_grows_target_dim_and_preserves_signs(self) -> None:
        emb = BAxUSEmbedding.create(input_dim=12, target_dim=2, seed=0)
        grown = emb.expand(new_bins_on_split=3)
        S_old, S_new = np.asarray(emb.matrix), np.asarray(grown.matrix)
        assert grown.target_dim == min(2 * 4, 12)
        assert grown.input_dim == 12
        # Every column keeps its sign and moves to exactly one (new) row.
        for col in range(12):
            assert np.count_nonzero(S_new[:, col]) == 1
            assert S_new[:, col].sum() == S_old[:, col].sum()

    def test_expand_caps_at_input_dim(self) -> None:
        emb = BAxUSEmbedding.create(input_dim=2, target_dim=1, seed=0)
        grown = emb.expand(new_bins_on_split=5)
        assert grown.target_dim == 2

    def test_round_trips_through_dump(self) -> None:
        emb = BAxUSEmbedding.create(input_dim=6, target_dim=2, seed=3)
        assert BAxUSEmbedding(**emb.model_dump()) == emb


class TestBAxUSUnsupportedOptions:
    """vocs-space point controls cannot be honored in an embedded subspace."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"fixed_features": {"x0": 0.5}},
            {"max_travel_distances": [0.1] * 6},
            {"n_interpolate_points": 3},
        ],
    )
    def test_rejected_at_construction(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError, match="BAxUS does not support"):
            BAxUSGenerator(vocs=_simple_vocs(), **kwargs)

    def test_custom_objective_rejected_at_construction(self) -> None:
        """custom_objective must pass its own CustomXoptObjective type validation
        before ours can fire, so it needs a real (minimal) subclass instance
        rather than a plain kwargs value."""
        vocs = _simple_vocs()
        objective = _MinimalCustomObjective(vocs=vocs)
        with pytest.raises(
            ValidationError, match="BAxUS does not support custom_objective"
        ):
            BAxUSGenerator(vocs=vocs, custom_objective=objective)

    def test_turbo_controller_rejected(self) -> None:
        with pytest.raises(
            ValidationError, match="no turbo controllers are compatible"
        ):
            BAxUSGenerator(vocs=_simple_vocs(), turbo_controller="optimize")

    @pytest.mark.parametrize(
        ("extra_variables", "vocs_kwargs", "match"),
        [
            pytest.param(
                {},
                {"constraints": {"c": ["LESS_THAN", 0.5]}},
                "does not support constraints",
                id="constraints",
            ),
            pytest.param(
                {"d": DiscreteVariable(values=[0.0, 1.0, 2.0])},
                {},
                "does not support discrete variables",
                id="discrete",
            ),
            pytest.param(
                {"c": ContextualVariable()},
                {},
                "does not support contextual variables",
                id="contextual",
            ),
        ],
    )
    def test_unsupported_vocs_rejected(
        self, extra_variables: dict[str, Any], vocs_kwargs: dict[str, Any], match: str
    ) -> None:
        """The target-space GP models only the objective over continuous
        vocs-space variables, so each of these must be rejected up front. Each
        case guards a supports_* = False override against the True default
        inherited from BayesianGenerator / ExpectedImprovementGenerator."""
        vocs = VOCS(
            variables={
                **{f"x{i}": [-1.0, 1.0] for i in range(3)},
                **extra_variables,
            },
            objectives={"f": "MAXIMIZE"},
            **vocs_kwargs,
        )
        with pytest.raises(VOCSError, match=match):
            BAxUSGenerator(vocs=vocs)


class TestBAxUSAcquisitionPinning:
    """Pin the acquisition contract: analytic LogEI on the target-space model
    with best_f = best weighted finite objective. Inherited from
    ExpectedImprovementGenerator, so these fail loudly if an xopt upgrade
    changes EI's acquisition semantics."""

    def test_analytic_logei_best_f_ignores_nan(self) -> None:
        gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=4), target_dim_init=2, seed=0)
        _seed_sobol(gen)
        point = gen.generate(1)[0]
        gen.add_data(pd.DataFrame([{**point, "f": float("nan")}]))

        acq = gen.get_acquisition(gen.train_model())

        assert isinstance(acq, LogExpectedImprovement)
        expected = np.nanmax(gen.data["f"].to_numpy(dtype=np.float64))
        assert float(cast(torch.Tensor, acq.best_f)) == pytest.approx(expected)

    def test_minimize_negates_best_f(self) -> None:
        vocs = VOCS(
            variables={f"x{i}": [-1.0, 1.0] for i in range(4)},
            objectives={"f": "MINIMIZE"},
        )
        gen = BAxUSGenerator(vocs=vocs, target_dim_init=2, seed=0)
        _seed_sobol(gen)

        acq = gen.get_acquisition(gen.train_model())

        assert isinstance(acq, LogExpectedImprovement)
        expected = -np.min(gen.data["f"].to_numpy(dtype=np.float64))
        assert float(cast(torch.Tensor, acq.best_f)) == pytest.approx(expected)


class TestBAxUSIsBayesianGenerator:
    def test_subclasses_bayesian_generator(self) -> None:
        assert isinstance(BAxUSGenerator(vocs=_simple_vocs()), BayesianGenerator)

    def test_subclasses_expected_improvement_generator(self) -> None:
        assert issubclass(BAxUSGenerator, ExpectedImprovementGenerator)

    def test_default_gp_constructor_is_standard(self) -> None:
        assert BAxUSGenerator(vocs=_simple_vocs()).gp_constructor.name == "standard"


class TestBAxUSSerialization:
    """Dump -> construct -> reattach data must resume the run.

    Resume contract: ``gen.model_dump()`` then pop ``"model"`` - xopt ignores
    the ``exclude`` argument, so a trained generator's dump carries the live
    botorch model (not yaml-safe) unless popped explicitly. ``computation_time``
    needs the same treatment for raw yaml: it is a live ``pd.DataFrame`` once a
    BO-phase ``generate()`` has run.

    Data is reattached with ``set_data`` (equivalently, by assigning
    ``gen.data``). Note this is what ``Xopt`` itself does for a ``StateOwner``,
    so the ordinary ``Xopt.from_yaml`` path resumes correctly too - see
    ``TestBAxUSCheckpointIntegrity``.
    """

    def test_dump_excludes_data_and_round_trips_components(self) -> None:
        gen = BAxUSGenerator(vocs=_simple_vocs(), seed=7)
        _seed_sobol(gen)

        dump = gen.model_dump()
        dump.pop("model", None)
        assert "data" not in dump

        gen2 = BAxUSGenerator(**dump)
        assert gen2.embedding == gen.embedding
        assert gen2.trust_region == gen.trust_region
        assert gen2.sobol_draws == gen.sobol_draws

    def test_dump_is_yaml_safe(self) -> None:
        gen = BAxUSGenerator(vocs=_simple_vocs(), seed=7)
        dump = gen.model_dump()
        dump.pop("model", None)
        mid_dump = BAxUSGenerator(**dump).model_dump()
        mid_dump.pop("model", None)
        reloaded = yaml.safe_load(yaml.safe_dump(mid_dump))
        assert BAxUSGenerator(**reloaded).embedding == gen.embedding

        # Post-BO a live botorch model and a populated computation_time frame
        # ride along; without the pops, yaml.safe_dump raises RepresenterError
        # on the trained ModelListGP / the pd.DataFrame.
        _seed_sobol(gen)
        _bo_step(gen)
        bo_dump = gen.model_dump()
        bo_dump.pop("model", None)
        bo_dump.pop("computation_time", None)
        yaml.safe_dump(bo_dump)  # must not raise

        gen2 = BAxUSGenerator(**bo_dump)
        assert gen2.embedding == gen.embedding
        assert gen2.trust_region == gen.trust_region

    def test_sobol_sequence_resumes_exactly(self) -> None:
        """Seed phase is bit-identical after a round trip (fast-forwarded engine)."""
        g1 = BAxUSGenerator(vocs=_simple_vocs(), seed=7)
        for _ in range(2):
            point = g1.generate(1)[0]
            g1.add_data(pd.DataFrame([{**point, **_sphere(point)}]))

        g2 = _round_trip(g1, with_data=True)

        assert g1.generate(1)[0] == g2.generate(1)[0]

    def test_bo_phase_state_identical_after_round_trip(self) -> None:
        """With the same torch seed, the resumed generator proposes the same point."""
        g1 = BAxUSGenerator(vocs=_simple_vocs(n_vars=4), target_dim_init=2, seed=0)
        _seed_sobol(g1)
        _bo_step(g1)

        g2 = _round_trip(g1, with_data=True)

        torch.manual_seed(0)
        p1 = g1.generate(1)[0]
        torch.manual_seed(0)
        p2 = g2.generate(1)[0]
        assert p1 == pytest.approx(p2)


class TestBAxUSRegistry:
    """The generator must be reachable through xopt's dynamic registry."""

    def test_get_generator_dynamic(self):
        assert get_generator_dynamic("baxus") is BAxUSGenerator

    def test_listed_in_available_generators(self):
        assert "baxus" in list_available_generators()


class TestBAxUSGetOptimum:
    """get_optimum must search the full embedded target space (not vocs-space
    bounds against the target-space model) and lift the result to vocs space."""

    def test_get_optimum_returns_single_row_in_bounds(self) -> None:
        vocs = _simple_vocs(n_vars=4)
        gen = BAxUSGenerator(vocs=vocs, target_dim_init=2, seed=0)

        _seed_sobol(gen)
        for _ in range(3):
            _bo_step(gen)

        opt = gen.get_optimum()

        assert isinstance(opt, pd.DataFrame)
        assert len(opt) == 1
        assert list(opt.columns) == vocs.variable_names
        _assert_point_in_bounds(opt.iloc[0].to_dict(), vocs)


class TestBAxUSVisualizeModel:
    """The inherited visualization is keyed to vocs-space variables and is
    meaningless (or crashes) against the embedded target-space model."""

    def test_visualize_model_raises_not_implemented(self) -> None:
        gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=4), target_dim_init=2, seed=0)

        with pytest.raises(NotImplementedError, match="target space"):
            gen.visualize_model()


class TestBAxUSFailureToleranceZeroDivision:
    """length_init == length_min makes the reference halvings formula 0,
    which must not divide by zero."""

    def test_equal_length_init_and_min_does_not_raise(self) -> None:
        gen = BAxUSGenerator(
            vocs=_simple_vocs(n_vars=8),
            target_dim_init=2,
            n_initial_sobol=2,
            eval_budget=50,
            length_init=0.5**7,
        )
        gen._refresh_failure_tolerance()
        assert gen.trust_region.failure_tolerance >= 1


class TestBAxUSOptimizationBounds:
    """The trust-region box itself.

    No end-to-end test can pin this down: replacing _get_optimization_bounds with
    the full domain, or centering it on the worst point, leaves every behavioral
    test passing. It needs direct assertions on the returned box.
    """

    @staticmethod
    def _gen_with_known_incumbent() -> tuple[BAxUSGenerator, int]:
        """Four points near the origin; row 2 is the unique best (MAXIMIZE)."""
        vocs = _simple_vocs(n_vars=4)
        gen = BAxUSGenerator(
            vocs=vocs, target_dim_init=2, n_initial_sobol=2, seed=0, eval_budget=50
        )
        rows = [
            {"x0": 0.10, "x1": -0.05, "x2": 0.00, "x3": 0.10, "f": -1.0},
            {"x0": -0.10, "x1": 0.05, "x2": 0.10, "x3": -0.05, "f": -2.0},
            {"x0": 0.05, "x1": 0.10, "x2": -0.05, "x3": 0.00, "f": 5.0},  # best
            {"x0": 0.00, "x1": 0.00, "x2": 0.05, "x3": 0.05, "f": -3.0},
        ]
        gen.add_data(pd.DataFrame(rows))
        return gen, 2

    def _incumbent_in_target_space(self, gen: BAxUSGenerator, row: int) -> torch.Tensor:
        Z = gen.embedding.project(gen._normalized_inputs(gen.data))
        return torch.tensor(Z[row], dtype=torch.double)

    def test_box_is_centered_on_the_incumbent(self) -> None:
        """argmin instead of argmax here would center on the worst point."""
        gen, best_row = self._gen_with_known_incumbent()
        gen.trust_region.length = 0.4

        bounds = gen._get_optimization_bounds()
        center = (bounds[0] + bounds[1]) / 2.0

        assert torch.allclose(
            center, self._incumbent_in_target_space(gen, best_row), atol=1e-12
        )

    def test_minimize_centers_on_the_smallest_objective(self) -> None:
        """The incumbent is direction-aware, not just the numeric maximum."""
        vocs = VOCS(
            variables={f"x{i}": [-1.0, 1.0] for i in range(4)},
            objectives={"f": "MINIMIZE"},
        )
        gen = BAxUSGenerator(
            vocs=vocs, target_dim_init=2, n_initial_sobol=2, seed=0, eval_budget=50
        )
        rows = [
            {"x0": 0.10, "x1": -0.05, "x2": 0.00, "x3": 0.10, "f": 4.0},
            {"x0": 0.05, "x1": 0.10, "x2": -0.05, "x3": 0.00, "f": -7.0},  # best
            {"x0": 0.00, "x1": 0.00, "x2": 0.05, "x3": 0.05, "f": 2.0},
        ]
        gen.add_data(pd.DataFrame(rows))
        gen.trust_region.length = 0.4

        bounds = gen._get_optimization_bounds()
        center = (bounds[0] + bounds[1]) / 2.0
        Z = gen.embedding.project(gen._normalized_inputs(gen.data))

        assert torch.allclose(
            center, torch.tensor(Z[1], dtype=torch.double), atol=1e-12
        )

    @pytest.mark.parametrize("length", [0.1, 0.25, 0.4])
    def test_box_width_is_the_trust_region_length(self, length: float) -> None:
        """In [-1, 1] target units the unclamped width is 2 * length (the length
        fields are in [0, 1] units, per the BoTorch reference)."""
        gen, _ = self._gen_with_known_incumbent()
        gen.trust_region.length = length

        bounds = gen._get_optimization_bounds()
        width = bounds[1] - bounds[0]

        assert torch.allclose(
            width, torch.full_like(width, 2.0 * length), atol=1e-12
        ), f"expected width {2.0 * length}, got {width.tolist()}"

    def test_box_stays_inside_the_domain(self) -> None:
        """A wide region around an incumbent at the edge must clamp, not overflow."""
        vocs = _simple_vocs(n_vars=4)
        gen = BAxUSGenerator(
            vocs=vocs, target_dim_init=2, n_initial_sobol=2, seed=0, eval_budget=50
        )
        gen.add_data(
            pd.DataFrame(
                [
                    {"x0": 1.0, "x1": 1.0, "x2": 1.0, "x3": 1.0, "f": 5.0},
                    {"x0": -1.0, "x1": -1.0, "x2": -1.0, "x3": -1.0, "f": -5.0},
                ]
            )
        )
        gen.trust_region.length = gen.trust_region.length_max

        bounds = gen._get_optimization_bounds()

        assert bool((bounds[0] >= -1.0).all())
        assert bool((bounds[1] <= 1.0).all())
        assert bool((bounds[1] > bounds[0]).all())

    def test_lengthscale_weights_stretch_the_box_per_dimension(self) -> None:
        """A dimension with a longer lengthscale gets a wider side."""
        gen, _ = self._gen_with_known_incumbent()
        gen.trust_region.length = 0.2

        uniform = gen._get_optimization_bounds()
        gen._lengthscale_weights = lambda model: torch.tensor(  # type: ignore[method-assign]
            [0.5, 2.0], dtype=torch.double
        )
        weighted = gen._get_optimization_bounds()

        uniform_width = uniform[1] - uniform[0]
        weighted_width = weighted[1] - weighted[0]
        assert weighted_width[0] < uniform_width[0]
        assert weighted_width[1] > uniform_width[1]


class TestBAxUSCandidateComesFromTheAcquisition:
    """The BO candidate must be the acquisition optimum inside the trust region.

    No objective-value comparison can establish this. On a quadratic, the center
    of a symmetric domain beats the average random point by exactly the variance
    of a random draw - for *any* placement of the optimum - so a generator that
    discarded the acquisition result and always proposed the center would still
    "beat Sobol on average". The candidate has to be checked against the box the
    acquisition was optimized over.
    """

    @staticmethod
    def _gen_with_offset_incumbent(
        z_star: np.ndarray | None = None,
    ) -> tuple[BAxUSGenerator, VOCS]:
        """Incumbent placed at a known, off-origin point of the target space."""
        vocs = _simple_vocs(n_vars=6)
        gen = BAxUSGenerator(
            vocs=vocs, target_dim_init=2, n_initial_sobol=2, seed=0, eval_budget=40
        )
        # lift an explicit target-space point so the incumbent projects back to it
        if z_star is None:
            z_star = np.array([[0.8, -0.7]])
        x_star = gen.embedding.lift(z_star)[0]
        best = {name: float(v) for name, v in zip(vocs.variable_names, x_star)}

        rng = np.random.default_rng(0)
        rows = [{**best, "f": 10.0}]
        for _ in range(5):
            other = {
                name: float(v)
                for name, v in zip(vocs.variable_names, rng.uniform(-1, 1, 6))
            }
            rows.append({**other, "f": -5.0})
        gen.add_data(pd.DataFrame(rows))
        return gen, vocs

    def test_candidate_lies_inside_the_trust_region(self) -> None:
        gen, vocs = self._gen_with_offset_incumbent()

        gen.generate(1)  # fold in results and fit the model
        gen.trust_region.length = 0.1  # narrow box, far from the origin
        point = gen.generate(1)[0]

        # no data was added, so this reproduces the box the candidate came from
        bounds = gen._get_optimization_bounds()
        lb, ub = bounds[0].numpy(), bounds[1].numpy()

        assert not ((lb <= 0.0).all() and (ub >= 0.0).all()), (
            "box contains the origin, so this test could not detect a constant proposal"
        )

        z = gen.embedding.project(gen._normalized_inputs(pd.DataFrame([point])))[0]
        assert (z >= lb - 1e-9).all(), f"candidate {z} below trust region {lb}"
        assert (z <= ub + 1e-9).all(), f"candidate {z} above trust region {ub}"

    def test_candidate_tracks_the_incumbent(self) -> None:
        """Move the incumbent, and the proposal must follow it.

        Lengthscale weights are pinned to 1 so the box is exactly ``length`` wide
        around the incumbent. Left to a real fit, a handful of points produces
        extreme ARD ratios that widen one side to the whole domain, and the
        proposal is then free to sit anywhere along it.
        """
        points = []
        for z_star in (np.array([[0.7, 0.7]]), np.array([[-0.7, -0.7]])):
            gen, _ = self._gen_with_offset_incumbent(z_star)

            gen.generate(1)
            gen.trust_region.length = 0.2
            gen._lengthscale_weights = lambda model: torch.ones(  # type: ignore[method-assign]
                2, dtype=torch.double
            )
            candidate = gen.generate(1)[0]
            points.append(
                gen.embedding.project(
                    gen._normalized_inputs(pd.DataFrame([candidate]))
                )[0]
            )

        assert points[0][0] > points[1][0], (
            f"proposal did not follow the incumbent: {points[0]} vs {points[1]}"
        )
        assert points[0][1] > points[1][1]


class TestBAxUSLengthscaleWeightNormalization:
    """The volume-preserving normalization must survive high target_dim.

    Forming the raw product before taking the root underflows to zero once
    target_dim reaches a few hundred, which makes every weight inf and silently
    widens the trust region to the entire domain.
    """

    @pytest.mark.parametrize("target_dim", [8, 128, 600, 1200])
    def test_weights_are_finite_and_volume_preserving(self, target_dim: int) -> None:
        torch.manual_seed(0)
        lengthscales = (
            torch.distributions.LogNormal(0.0, 2.0).sample((target_dim,)).double()
        )

        weights = _normalize_lengthscale_weights(lengthscales, target_dim)

        assert bool(torch.isfinite(weights).all()), "weights underflowed to inf"
        assert bool((weights > 0).all())
        # unit geometric mean
        assert float(weights.log().mean().exp()) == pytest.approx(1.0, rel=1e-9)

    def test_ordering_is_preserved(self) -> None:
        lengthscales = torch.tensor([0.5, 4.0, 1.0], dtype=torch.double)
        weights = _normalize_lengthscale_weights(lengthscales, 3)
        assert weights.argsort().tolist() == lengthscales.argsort().tolist()


class TestBAxUSExactScheduleConstants:
    """The trust region halves and doubles by exactly a factor of two.

    Asserting only the direction (got smaller / got larger) leaves the schedule
    free to drift, which changes how many failures it takes to trigger an
    embedding expansion.
    """

    def test_failure_halves_the_length_exactly(self) -> None:
        tr = BAxUSTrustRegion(target_dim=4, failure_tolerance=1, length=0.8)
        tr.best_value = 10.0
        tr.update(torch.tensor([1.0]))  # no improvement
        assert tr.length == pytest.approx(0.4, rel=1e-12)

    def test_success_doubles_the_length_exactly(self) -> None:
        tr = BAxUSTrustRegion(target_dim=4, success_tolerance=1, length=0.4)
        tr.best_value = 0.0
        tr.update(torch.tensor([5.0]))  # improvement
        assert tr.length == pytest.approx(0.8, rel=1e-12)

    def test_growth_is_capped_at_length_max(self) -> None:
        tr = BAxUSTrustRegion(
            target_dim=4, success_tolerance=1, length=1.0, length_max=1.6
        )
        tr.best_value = 0.0
        for step in range(3):
            tr.update(torch.tensor([10.0 * (step + 1)]))
        assert tr.length == pytest.approx(1.6, rel=1e-12)

    def test_improvement_needs_to_clear_the_tolerance(self) -> None:
        """A gain below 0.1% of the incumbent counts as a failure."""
        tr = BAxUSTrustRegion(target_dim=4, failure_tolerance=1, length=0.8)
        tr.best_value = 100.0
        tr.update(torch.tensor([100.05]))  # +0.05% -> not an improvement
        assert tr.failure_counter == 0  # reset after the shrink fired
        assert tr.length == pytest.approx(0.4, rel=1e-12)

        tr2 = BAxUSTrustRegion(target_dim=4, failure_tolerance=1, length=0.8)
        tr2.best_value = 100.0
        tr2.update(torch.tensor([100.5]))  # +0.5% -> an improvement
        assert tr2.length == pytest.approx(0.8, rel=1e-12)
        assert tr2.success_counter == 1


class TestBAxUSSobolDomain:
    """Seed points must cover the whole target cube, not just one orthant."""

    def test_sobol_draws_span_negative_and_positive(self) -> None:
        gen = BAxUSGenerator(
            vocs=_simple_vocs(n_vars=6),
            target_dim_init=2,
            n_initial_sobol=32,
            seed=0,
            eval_budget=60,
        )
        zs = np.vstack([gen._draw_sobol_point() for _ in range(32)])

        assert zs.min() < -0.2, f"no negative seed coordinates (min {zs.min():.3f})"
        assert zs.max() > 0.2, f"no positive seed coordinates (max {zs.max():.3f})"
        assert (zs >= -1.0).all() and (zs <= 1.0).all()
        assert abs(float(zs.mean())) < 0.25, "seed points are not centered"


class TestBAxUSExpansionRandomization:
    """Splitting must randomize which dimensions land in which sub-bin.

    The reference permutes the contributing dimensions before splitting. Without
    it the split is a fixed function of column index, so two dimensions that are
    adjacent in index and share a bin are never separated until the embedding
    reaches full dimensionality - which is exactly where BAxUS has no advantage
    left. That failure is invisible to structural checks: the matrix is still a
    valid partition either way.
    """

    INPUT_DIM = 64
    B = 3

    @staticmethod
    def _bin_of(emb: BAxUSEmbedding, col: int) -> int:
        matrix = np.asarray(emb.matrix)
        return int(np.nonzero(matrix[:, col])[0][0])

    def _collision_rate(self, target_dim: int, trials: int = 200) -> float:
        """How often input dims 0 and 1 (adjacent in index) share a bin."""
        collisions = 0
        for trial in range(trials):
            emb = BAxUSEmbedding.create(
                input_dim=self.INPUT_DIM, target_dim=2, seed=trial
            )
            while emb.target_dim < target_dim:
                emb = emb.expand(self.B, seed=trial * 7919 + emb.target_dim)
            collisions += self._bin_of(emb, 0) == self._bin_of(emb, 1)
        return collisions / trials

    def test_expansion_separates_adjacent_dimensions(self) -> None:
        """Collision rate must fall as the embedding grows.

        A deterministic split leaves it pinned at P(same initial bin) = 1/2.
        """
        rate_8 = self._collision_rate(target_dim=8)
        rate_32 = self._collision_rate(target_dim=32)

        assert rate_8 < 0.30, f"dims 0/1 still collide {rate_8:.0%} of the time at d=8"
        assert rate_32 < rate_8, (
            f"expanding did not decorrelate the split: {rate_8:.3f} -> {rate_32:.3f}"
        )

    def test_expand_is_reproducible_for_a_given_seed(self) -> None:
        emb = BAxUSEmbedding.create(input_dim=self.INPUT_DIM, target_dim=2, seed=0)
        assert emb.expand(self.B, seed=7) == emb.expand(self.B, seed=7)

    def test_different_seeds_give_different_partitions(self) -> None:
        emb = BAxUSEmbedding.create(input_dim=self.INPUT_DIM, target_dim=2, seed=0)
        assert emb.expand(self.B, seed=1) != emb.expand(self.B, seed=2)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_randomized_expansion_keeps_the_matrix_well_formed(self, seed: int) -> None:
        """Every input dimension stays assigned to exactly one bin, with its sign."""
        emb = BAxUSEmbedding.create(input_dim=self.INPUT_DIM, target_dim=2, seed=seed)
        original = np.asarray(emb.matrix)
        original_sign = {
            col: original[:, col][np.nonzero(original[:, col])[0][0]]
            for col in range(self.INPUT_DIM)
        }

        while emb.target_dim < self.INPUT_DIM:
            emb = emb.expand(self.B, seed=seed * 31 + emb.target_dim)
            matrix = np.asarray(emb.matrix)

            assert (np.count_nonzero(matrix, axis=0) == 1).all(), (
                "column lost/duplicated"
            )
            assert (np.count_nonzero(matrix, axis=1) > 0).all(), "empty bin"
            for col in range(self.INPUT_DIM):
                nz = np.nonzero(matrix[:, col])[0][0]
                assert matrix[nz, col] == original_sign[col], "sign changed on split"

        # at full dimensionality the embedding is a signed permutation
        assert emb.target_dim == self.INPUT_DIM
        assert (np.count_nonzero(np.asarray(emb.matrix), axis=1) == 1).all()

    def test_generator_expansion_survives_a_round_trip(self) -> None:
        """The expansion seed is derived from (seed, n_expansions), so a run
        restored mid-flight expands to the same embedding."""
        vocs = _simple_vocs(n_vars=16)
        gen = BAxUSGenerator(vocs=vocs, target_dim_init=2, seed=3, eval_budget=60)
        _seed_sobol(gen)

        restored = _round_trip(gen, with_data=True)

        for generator in (gen, restored):
            _force_expansion(generator)
            generator.add_data(
                pd.DataFrame([{**{f"x{i}": 0.1 for i in range(16)}, "f": -1.0}])
            )
            generator.generate(1)

        assert gen.n_expansions == 1
        assert restored.embedding == gen.embedding


class TestBAxUSCheckpointIntegrity:
    """Saving and reloading through Xopt must not advance the algorithm.

    Xopt's data validator hands a restored dataframe to the generator; for a
    generator that owns state, that must not replay the history it already
    reflects. BAxUS is a StateOwner precisely so this is a no-op.
    """

    @staticmethod
    def _state(gen: BAxUSGenerator) -> dict[str, Any]:
        tr = gen.trust_region
        return {
            "target_dim": gen.embedding.target_dim,
            "matrix": gen.embedding.matrix,
            "length": tr.length,
            "success_counter": tr.success_counter,
            "failure_counter": tr.failure_counter,
            "failure_tolerance": tr.failure_tolerance,
            "best_value": tr.best_value,
            "sobol_draws": gen.sobol_draws,
            "n_expansions": gen.n_expansions,
            "tr_observed_rows": gen.tr_observed_rows,
        }

    def _run(self, steps: int = 12) -> Xopt:
        vocs = _simple_vocs(n_vars=6)
        X = Xopt(
            evaluator=XoptEvaluator(function=_sphere),
            generator=BAxUSGenerator(
                vocs=vocs, target_dim_init=2, seed=0, eval_budget=40
            ),
        )
        for _ in range(steps):
            X.step()
        return X

    def test_is_a_state_owner(self) -> None:
        assert isinstance(BAxUSGenerator(vocs=_simple_vocs(), seed=0), StateOwner)

    def test_repeated_checkpoint_cycles_do_not_move_the_trust_region(self) -> None:
        """No evaluations happen between cycles, so nothing may change."""
        X = self._run()
        expected = self._state(X.generator)
        assert len(X.data) == 12

        for cycle in range(6):
            X = Xopt.from_yaml(X.yaml())
            assert len(X.data) == 12, f"data changed on cycle {cycle}"
            assert self._state(X.generator) == expected, (
                f"generator state drifted on checkpoint cycle {cycle}"
            )

    def test_resume_folds_in_only_the_new_results(self) -> None:
        """The pre-restore history must not be re-ingested.

        A replay would re-consume everything from the seed quota onward; a
        correct resume advances the trust region by exactly the one row that was
        evaluated after the checkpoint.
        """
        X = self._run()
        before = self._state(X.generator)

        resumed = Xopt.from_yaml(X.yaml())
        resumed.step()
        after = self._state(resumed.generator)

        assert after["tr_observed_rows"] == before["tr_observed_rows"] + 1
        assert after["n_expansions"] == before["n_expansions"]
        # one tick only: at most one counter moved, by one
        assert abs(after["failure_counter"] - before["failure_counter"]) <= 1
        assert abs(after["success_counter"] - before["success_counter"]) <= 1

    def test_resumed_run_continues_the_same_trajectory(self) -> None:
        """One more step on the original and on the restored run agree.

        The proposed point is not compared exactly: xopt's yaml writer keeps 10
        significant digits, and that ~1e-10 perturbation of the training data is
        amplified by GP hyperparameter fitting and the L-BFGS acquisition
        optimization. The algorithm state is insensitive to it.
        """
        X = self._run()
        resumed = Xopt.from_yaml(X.yaml())

        for runner in (X, resumed):
            runner.step()

        original, restored = self._state(X.generator), self._state(resumed.generator)
        for key in (
            "target_dim",
            "matrix",
            "length",
            "success_counter",
            "failure_counter",
            "failure_tolerance",
            "sobol_draws",
            "n_expansions",
            "tr_observed_rows",
        ):
            assert original[key] == restored[key], f"{key} diverged after resume"
        assert original["best_value"] == pytest.approx(restored["best_value"], rel=1e-6)


class TestBAxUSFailedEvaluations:
    """An evaluator that raises returns a row with no objective column at all."""

    def test_add_data_tolerates_a_missing_objective_column(self) -> None:
        gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=4), target_dim_init=2, seed=0)

        failed = {f"x{i}": 0.1 for i in range(4)}
        failed["xopt_error"] = True
        gen.add_data(pd.DataFrame([failed]))

        assert len(gen.data) == 1
        assert gen._finite_data().empty

    def test_xopt_run_survives_a_failing_evaluator(self) -> None:
        """The generator must not be the reason a non-strict run dies."""
        vocs = _simple_vocs(n_vars=4)
        calls = {"n": 0}

        def flaky(inputs: dict[str, float]) -> dict[str, float]:
            calls["n"] += 1
            if calls["n"] == 4:
                raise RuntimeError("evaluation blew up")
            return _sphere(inputs)

        X = Xopt(
            evaluator=XoptEvaluator(function=flaky),
            generator=BAxUSGenerator(
                vocs=vocs, target_dim_init=2, seed=0, eval_budget=30
            ),
            strict=False,
        )
        for _ in range(8):
            X.step()

        assert len(X.data) == 8
        assert len(X.generator._finite_data()) == 7  # the failed row is excluded


class TestBAxUSObservables:
    """The target-space model has a single outcome while the inherited
    acquisition scalarizes over every vocs output, so observables cannot work."""

    def test_observables_rejected_at_construction(self) -> None:
        vocs = VOCS(
            variables={f"x{i}": [-1.0, 1.0] for i in range(4)},
            objectives={"f": "MAXIMIZE"},
            observables=["g"],
        )
        with pytest.raises(ValidationError, match="does not support observables"):
            BAxUSGenerator(vocs=vocs, target_dim_init=2, seed=0)


class TestBAxUSEvalBudgetWarning:
    """Without eval_budget the reference expansion schedule is unavailable, so
    construction warns. It must warn exactly once: the generator assigns to its
    own fields on every BO step, and a model validator of either mode re-fires
    on assignment under validate_assignment, which would bury the run in
    duplicate warnings."""

    def test_warns_once_on_construction(self) -> None:
        with pytest.warns(GeneratorWarning, match="eval_budget is not set") as caught:
            gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=4), target_dim_init=2, seed=0)
        assert len(caught) == 1

        with warnings.catch_warnings(record=True) as on_assignment:
            warnings.simplefilter("always")
            gen.sobol_draws = 3
            gen.n_expansions = 1
        assert [w for w in on_assignment if w.category is GeneratorWarning] == []

    def test_no_warning_when_budget_is_set(self) -> None:
        vocs = _simple_vocs(n_vars=4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            BAxUSGenerator(vocs=vocs, target_dim_init=2, seed=0, eval_budget=50)
        assert [w for w in caught if w.category is GeneratorWarning] == []


class TestBAxUSComputationTime:
    """generate() must record training/acquisition-optimization timing during
    the BO phase, matching the base class's bookkeeping."""

    def test_bo_generate_records_one_row(self) -> None:
        gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=4), target_dim_init=2, seed=0)
        _seed_sobol(gen)

        assert gen.computation_time is None
        _bo_step(gen)

        assert isinstance(gen.computation_time, pd.DataFrame)
        assert len(gen.computation_time) == 1
        assert list(gen.computation_time.columns) == [
            "training",
            "acquisition_optimization",
        ]

    def test_second_bo_generate_appends_row(self) -> None:
        gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=4), target_dim_init=2, seed=0)
        _seed_sobol(gen)

        _bo_step(gen)
        _bo_step(gen)

        assert len(gen.computation_time) == 2
