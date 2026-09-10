import math
import warnings
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
from xopt.generators import get_generator_dynamic
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


# most tests here construct generators without eval_budget on purpose, so the
# advisory would otherwise fire on nearly every construction; the contract itself
# stays pinned by test_missing_eval_budget_warns_exactly_once, whose own filters
# override this one
@pytest.fixture(autouse=True)
def _quiet_missing_eval_budget():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="BAxUS: eval_budget is not set",
            category=GeneratorWarning,
        )
        yield


def _simple_vocs(n_vars: int = 6) -> VOCS:
    variables = {f"x{i}": [-1.0, 1.0] for i in range(n_vars)}
    return VOCS(variables=variables, objectives={"f": "MAXIMIZE"})


def _sphere(inputs: dict[str, float]) -> dict[str, float]:
    # negative sphere - maximum at the origin
    return {"f": -sum(v**2 for v in inputs.values())}


def _assert_point_in_bounds(point: dict[str, float], vocs: VOCS) -> None:
    for name in vocs.variable_names:
        lo, hi = vocs.variables[name].domain
        assert lo <= point[name] <= hi, f"{name}={point[name]} outside [{lo}, {hi}]"


# CustomXoptObjective is abstract, so the rejection path needs a real subclass
class _MinimalCustomObjective(CustomXoptObjective):
    def forward(
        self, samples: torch.Tensor, X: torch.Tensor | None = None
    ) -> torch.Tensor:
        return samples


def _vocs_with(**vocs_kwargs: Any) -> dict:
    # ``vocs=`` kwargs for a minimal VOCS carrying one unsupported feature
    return {
        "vocs": VOCS(
            variables={f"x{i}": [-1.0, 1.0] for i in range(3)},
            objectives={"f": "MAXIMIZE"},
            **vocs_kwargs,
        )
    }


def _bare_gen() -> BAxUSGenerator:
    return BAxUSGenerator(vocs=_simple_vocs(n_vars=4), target_dim_init=2, seed=0)


def _seed_sobol(gen: BAxUSGenerator) -> None:
    for _ in range(gen.n_initial_sobol):
        point = gen.generate(1)[0]
        gen.add_data(pd.DataFrame([{**point, **_sphere(point)}]))


def _force_expansion(gen: BAxUSGenerator) -> None:
    # below length_min with an unbeatable incumbent, so the next fold-in trips
    # the restart -> expand branch
    gen.trust_region.length = gen.trust_region.length_min / 2.0
    gen.trust_region.best_value = 1e9  # guarantee a failure (no improvement)


def _round_trip(gen: BAxUSGenerator, *, with_data: bool = False) -> BAxUSGenerator:
    # the documented resume path: the live botorch model is not serializable and
    # is popped by hand, exactly as the docs instruct
    dump = gen.model_dump()
    dump.pop("model", None)
    restored = BAxUSGenerator(**dump)
    if with_data and gen.data is not None:
        restored.data = gen.data.copy()
    return restored


def _gen_with_target_space_data(
    z_rows: list[list[float]],
    ys: list[float],
    *,
    n_vars: int = 4,
    direction: str = "MAXIMIZE",
) -> BAxUSGenerator:
    # rows are given in target space and lifted into vocs space, so each one
    # projects back to exactly the z_rows entry it came from - which is what makes
    # "the box is centered on *this* point" assertable. ys is in weighted form
    # (higher is better) and is mirrored for MINIMIZE, so the same row stays the
    # incumbent in either direction.
    vocs = VOCS(
        variables={f"x{i}": [-1.0, 1.0] for i in range(n_vars)},
        objectives={"f": direction},
    )
    gen = BAxUSGenerator(
        vocs=vocs, target_dim_init=2, n_initial_sobol=2, seed=0, eval_budget=50
    )
    sign = 1.0 if direction == "MAXIMIZE" else -1.0
    X = gen.embedding.lift(np.array(z_rows))
    gen.add_data(
        pd.DataFrame(
            [
                {
                    **{name: float(v) for name, v in zip(vocs.variable_names, x)},
                    "f": sign * y,
                }
                for x, y in zip(X, ys)
            ]
        )
    )
    return gen


def _in_target_space(gen: BAxUSGenerator, data: pd.DataFrame) -> np.ndarray:
    return gen.embedding.project(gen._normalized_inputs(data))


def _set_target_dim(gen: BAxUSGenerator, target_dim: int) -> None:
    gen.embedding = BAxUSEmbedding.create(
        input_dim=gen.embedding.input_dim, target_dim=target_dim, seed=0
    )
    gen.trust_region = BAxUSTrustRegion()


class TestBAxUSConstruction:
    @pytest.mark.parametrize(
        ("n_vars", "kwargs", "expected"),
        [(6, {}, 2), (10, {"target_dim_init": 4}, 4), (3, {"target_dim_init": 10}, 3)],
        ids=["defaults-to-2", "honors-the-knob", "capped-at-input-dim"],
    )
    def test_embedding_sized_from_vocs(
        self, n_vars: int, kwargs: dict[str, Any], expected: int
    ) -> None:
        gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=n_vars), **kwargs)

        assert gen.embedding.target_dim == expected
        assert gen.embedding.input_dim == n_vars

    # without vocs, _init_components must skip auto-creation and let pydantic
    # report the missing fields rather than crashing internally; a budget below
    # the seed quota would leave no BO phase at all; generate(2) must raise
    # rather than silently under-deliver one point; and the inherited
    # visualization is keyed to vocs-space variables
    @pytest.mark.parametrize(
        ("action", "error", "match"),
        [
            (BAxUSGenerator, ValidationError, "vocs"),
            (
                lambda: BAxUSGenerator(
                    vocs=_simple_vocs(), n_initial_sobol=10, eval_budget=5
                ),
                ValidationError,
                "must cover the seed quota",
            ),
            (
                lambda: _bare_gen().generate(2),
                NotImplementedError,
                "parallel candidate",
            ),
            (
                lambda: _bare_gen().visualize_model(),
                NotImplementedError,
                "target space",
            ),
        ],
        ids=[
            "missing-vocs",
            "budget-below-seed-quota",
            "batch-generation",
            "visualize-model",
        ],
    )
    def test_hard_refusals(
        self, action: Any, error: type[Exception], match: str
    ) -> None:
        with pytest.raises(error, match=match):
            action()

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        # an explicit quota wins; otherwise it is max(2, target_dim + 1)
        [
            ({"target_dim_init": 2, "n_initial_sobol": 5}, 5),
            ({"target_dim_init": 3}, 4),
        ],
        ids=["explicit", "derived"],
    )
    def test_n_initial_sobol_is_explicit_or_derived(
        self, kwargs: dict[str, Any], expected: int
    ) -> None:
        gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=6), **kwargs)
        assert gen.n_initial_sobol == expected

    def test_registry_bases_and_gp_constructor_default(self) -> None:
        assert get_generator_dynamic("baxus") is BAxUSGenerator
        assert issubclass(BAxUSGenerator, ExpectedImprovementGenerator)

        gen = BAxUSGenerator(vocs=_simple_vocs())
        assert gen.gp_constructor.name == "standard"
        # the near-interpolating GP the reference trust-region logic assumes
        assert gen.gp_constructor.use_low_noise_prior is True

    def test_missing_eval_budget_warns_exactly_once(self) -> None:
        # the generator assigns to its own fields on every BO step, and a model
        # validator of either mode re-fires on assignment under
        # validate_assignment, which would bury the run in duplicate warnings
        with pytest.warns(GeneratorWarning, match="eval_budget is not set") as caught:
            gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=4), target_dim_init=2, seed=0)
        assert len(caught) == 1

        with warnings.catch_warnings(record=True) as on_assignment:
            warnings.simplefilter("always")
            gen.sobol_draws = 3
            gen.n_expansions = 1
        assert [w for w in on_assignment if w.category is GeneratorWarning] == []

        with warnings.catch_warnings(record=True) as with_budget:
            warnings.simplefilter("always")
            BAxUSGenerator(
                vocs=_simple_vocs(n_vars=4), target_dim_init=2, seed=0, eval_budget=50
            )
        assert [w for w in with_budget if w.category is GeneratorWarning] == []


# seed pins the embedding and the Sobol initialisation; GP fitting and
# acquisition use torch's global RNG and are out of scope
class TestBAxUSSeeding:
    def test_seed_pins_the_embedding_and_the_sobol_sequence(self) -> None:
        # the Sobol phase must be one reproducible sequence - a fresh draw(1)
        # engine per call would instead repeat the same first point every time
        vocs = _simple_vocs(n_vars=6)
        g1 = BAxUSGenerator(vocs=vocs, seed=7)
        g2 = BAxUSGenerator(vocs=vocs, seed=7)
        g3 = BAxUSGenerator(vocs=vocs, seed=1)

        assert g1.embedding.matrix == g2.embedding.matrix
        assert g1.embedding.matrix != g3.embedding.matrix

        pts1 = [g1.generate(1)[0] for _ in range(g1.n_initial_sobol)]
        pts2 = [g2.generate(1)[0] for _ in range(g2.n_initial_sobol)]

        assert pts1 == pts2
        distinct = {tuple(p.values()) for p in pts1}
        assert len(distinct) == len(pts1)

    def test_raw_draws_span_the_cube_and_follow_the_target_dim(self) -> None:
        gen = BAxUSGenerator(
            vocs=_simple_vocs(n_vars=8),
            target_dim_init=2,
            n_initial_sobol=32,
            seed=0,
            eval_budget=60,
        )
        zs = np.vstack([gen._draw_sobol_point() for _ in range(32)])

        # the draws must cover the whole cube, not just one orthant
        assert zs.shape == (32, 2)
        assert zs.min() < -0.2, f"no negative seed coordinates (min {zs.min():.3f})"
        assert zs.max() > 0.2, f"no positive seed coordinates (max {zs.max():.3f})"
        assert (zs >= -1.0).all() and (zs <= 1.0).all()
        assert abs(float(zs.mean())) < 0.25, "seed points are not centered"

        # the engine is cached, so it has to be rebuilt when the embedding grows
        # under it; without the rebuild the stale engine feeds a wrongly-sized
        # vector into the lift and numpy raises on the matmul
        _set_target_dim(gen, 8)

        assert gen._draw_sobol_point().shape == (1, 8)


class TestBAxUSEndToEnd:
    def test_xopt_run_survives_failures_and_stays_in_bounds(self) -> None:
        # asymmetric per-variable bounds guard against the v2->v3 vocs.bounds
        # orientation flip (was (2, D), now (D, 2)): a wrong orientation produces
        # out-of-box points in the BO phase. get_optimum is checked for the same
        # reason - it optimizes in target space and lifts, so bounds at different
        # scales are what make a broken lift visible. The run is warm-started
        # with random points (like the real phases.py flow) rather than the
        # generator's own Sobol phase, so those rows reach the model by
        # least-squares projection, and the evaluator fails once mid-run: the
        # failed row stays in the history, is excluded from training, and later
        # BO steps still propose in-bounds points.
        vocs = VOCS(
            variables={"x0": [0.0, 10.0], "x1": [-165.0, 165.0], "x2": [0.5, 0.6]},
            objectives={"f": "MAXIMIZE"},
        )
        calls = {"n": 0}

        def flaky(inputs: dict[str, float]) -> dict[str, float]:
            calls["n"] += 1
            if calls["n"] == 6:  # the third BO step, after the 3 warm-start rows
                raise RuntimeError("evaluation blew up")
            return _sphere(inputs)

        gen = BAxUSGenerator(vocs=vocs, target_dim_init=2, seed=0, eval_budget=30)
        X = Xopt(evaluator=XoptEvaluator(function=flaky), generator=gen, strict=False)

        X.random_evaluate(gen.n_initial_sobol)
        assert X.generator.computation_time is None  # no model trained yet
        for _ in range(8):
            X.step()

        assert len(X.data) == gen.n_initial_sobol + 8
        # the failed row is kept but excluded from training
        assert len(X.generator._finite_data()) == len(X.data) - 1
        for _, row in X.data.iterrows():
            _assert_point_in_bounds(row.to_dict(), vocs)
        timings = X.generator.computation_time
        assert len(timings) == 8
        assert list(timings.columns) == ["training", "acquisition_optimization"]

        opt = X.generator.get_optimum()
        assert len(opt) == 1
        assert list(opt.columns) == vocs.variable_names
        _assert_point_in_bounds(opt.iloc[0].to_dict(), vocs)


class TestBAxUSTrainingData:
    def test_unusable_rows_are_kept_but_excluded_from_training(self) -> None:
        gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=4), target_dim_init=2)

        with pytest.raises(ValueError, match="no data available to build model"):
            gen.train_model()

        point = gen.generate(1)[0]
        gen.add_data(pd.DataFrame([{**point, "f": float("nan")}]))
        assert len(gen.data) == 1
        assert gen._finite_data().empty

        rows = []
        for y in (1.0, float("nan"), 2.0):
            point = gen.generate(1)[0]
            rows.append({**point, "f": y})
        gen.add_data(pd.DataFrame(rows))
        assert len(gen.data) == 4
        assert len(gen._finite_data()) == 2

        # an evaluator that raises returns a row with no objective column at all
        gen.add_data(
            pd.DataFrame([{**{f"x{i}": 0.1 for i in range(4)}, "xopt_error": True}])
        )
        assert len(gen.data) == 5
        assert len(gen._finite_data()) == 2


# trust-region state bookkeeping (weighted convention: higher is better), then
# which evaluations the generator folds into it - BO-phase only, since the
# reference starts BO with a pristine region
class TestBAxUSTrustRegion:
    def test_success_path_doubles_the_length_then_caps_it(self) -> None:
        # asserting only the direction (got larger) would leave the schedule free
        # to drift, which changes how many failures trigger an expansion
        tr = BAxUSTrustRegion(length=0.2)
        tr.best_value = 0.0
        improving = iter([100.0 * i for i in range(1, 100)])

        for _ in range(tr.success_tolerance - 1):
            tr.update(torch.tensor([next(improving)]), failure_tolerance=1)
        assert tr.length == pytest.approx(0.2, rel=1e-12)  # tolerance not reached

        tr.update(torch.tensor([next(improving)]), failure_tolerance=1)
        assert tr.length == pytest.approx(0.4, rel=1e-12)  # exactly doubled

        # 0.4 -> 0.8 -> 1.6, then pinned: a third block would reach 3.2 uncapped
        for _ in range(3 * tr.success_tolerance):
            tr.update(torch.tensor([next(improving)]), failure_tolerance=1)
        assert tr.length == pytest.approx(tr.length_max, rel=1e-12)

    def test_failure_path_halves_the_length_then_restarts(self) -> None:
        # the exact halving schedule is pinned by the gain-0.05%-fails case below
        tr = BAxUSTrustRegion(length=0.8)
        tr.best_value = 999.0

        for _ in range(100):
            if tr.restart_triggered:
                break
            tr.update(torch.tensor([-999.0]), failure_tolerance=1)
        assert tr.restart_triggered
        assert tr.length < tr.length_min

    # a gain below 0.1% of the incumbent counts as a failure, and a pristine
    # region always counts the first batch as a success
    @pytest.mark.parametrize(
        ("best_value", "y", "success_counter", "length"),
        [(None, -5.0, 1, 0.8), (100.0, 100.05, 0, 0.4), (100.0, 100.5, 1, 0.8)],
        ids=["pristine-always-improves", "gain-0.05%-fails", "gain-0.5%-improves"],
    )
    def test_improvement_needs_to_clear_the_tolerance(
        self, best_value: float | None, y: float, success_counter: int, length: float
    ) -> None:
        # success_counter and length are what tell the branches apart: on the
        # failure branch best_value is still assigned (best is None short-circuits
        # the max) and failure_counter is reset by the shrink it triggers
        tr = BAxUSTrustRegion(length=0.8)
        tr.best_value = best_value

        tr.update(torch.tensor([y]), failure_tolerance=1)

        assert tr.success_counter == success_counter
        assert tr.failure_counter == 0  # a shrink resets it either way
        assert tr.length == pytest.approx(length, rel=1e-12)
        assert tr.best_value == max(y, best_value if best_value is not None else y)

    @pytest.mark.parametrize("direction", ["MAXIMIZE", "MINIMIZE"])
    def test_seed_rows_never_fold_in_and_bo_results_fold_in_exactly_once(
        self, direction: str
    ) -> None:
        # the seed quota is what protects the pristine state, not the early
        # return, so the advance has to be driven directly. tr_observed_rows is
        # what makes the fold-in once-only: without it every advance re-folds the
        # whole BO history, which is silent while results keep arriving but ticks
        # the counters on an advance with no new results.
        vocs = VOCS(
            variables={f"x{i}": [-1.0, 1.0] for i in range(4)},
            objectives={"f": direction},
        )
        gen = BAxUSGenerator(vocs=vocs, target_dim_init=2, n_initial_sobol=2, seed=0)
        sign = 1.0 if direction == "MAXIMIZE" else -1.0
        row = {f"x{i}": 0.1 for i in range(4)}

        # seed rows are given the better weighted value (+100), so a fold-in of
        # the seed quota would surface below as best_value == 100.0
        gen.add_data(pd.DataFrame([{**row, "f": sign * 100.0}] * 2))
        gen._advance_trust_region()

        assert gen.trust_region.best_value is None
        assert gen.tr_observed_rows == 0

        gen.add_data(pd.DataFrame([{**row, "f": sign * 1.0}]))
        gen._advance_trust_region()

        assert gen.trust_region.best_value == 1.0  # only the post-quota row counts
        assert gen.tr_observed_rows == len(gen.data)

        tr = gen.trust_region
        before = (tr.length, tr.success_counter, tr.failure_counter, tr.best_value)
        observed = gen.tr_observed_rows

        gen._advance_trust_region()
        gen._advance_trust_region()

        tr = gen.trust_region
        assert (
            tr.length,
            tr.success_counter,
            tr.failure_counter,
            tr.best_value,
        ) == before
        assert gen.tr_observed_rows == observed


# budget-aware failure tolerance (paper Alg. 1). input_dim=16 with b=3 is an
# exact power of b+1, where the BoTorch tutorial's BaxusState derives the same
# d_init (=1) and split count (=2) we configure - the regime in which a verbatim
# comparison against the reference is meaningful.
class TestBAxUSFailureTolerance:
    def _gen(self, eval_budget: int | None, target_dim_init: int = 1) -> BAxUSGenerator:
        return BAxUSGenerator(
            vocs=_simple_vocs(n_vars=16),
            target_dim_init=target_dim_init,
            n_initial_sobol=10,
            seed=0,
            eval_budget=eval_budget,
        )

    # expected values are the tutorial's BaxusState.failure_tolerance at dim=16,
    # b=3, d_init=1, with eval_budget as the total including the 10 seed points.
    # (103, 8) is the discriminating pair: computing the budget without
    # subtracting the seeds gives 6 instead of 5.
    @pytest.mark.parametrize(
        ("eval_budget", "target_dim", "expected"),
        [(40, 4, 1), (40, 8, 1), (103, 4, 3), (103, 8, 5), (210, 4, 4), (210, 8, 8)],
    )
    def test_matches_the_reference_schedule(
        self, eval_budget: int, target_dim: int, expected: int
    ) -> None:
        gen = self._gen(eval_budget=eval_budget)
        _set_target_dim(gen, target_dim)
        assert gen._failure_tolerance() == expected

    def test_branch_selection(self) -> None:
        # without a budget: the ceil(target_dim / 2) heuristic
        heuristic = self._gen(eval_budget=None, target_dim_init=4)
        assert heuristic._failure_tolerance() == 2  # ceil(4/2)

        # at full dimensionality: target_dim, budget or not
        full_dim = self._gen(eval_budget=None, target_dim_init=16)
        assert full_dim._failure_tolerance() == 16

        # length_init == length_min makes the reference halvings formula 0, which
        # must not divide by zero: it is floored to 1, so the budget share is
        # spent in one go - round(3 * 48 * 2 / (2 * 15)) = 10, capped at target_dim
        degenerate = BAxUSGenerator(
            vocs=_simple_vocs(n_vars=8),
            target_dim_init=2,
            n_initial_sobol=2,
            eval_budget=50,
            length_init=0.5**7,
        )
        assert degenerate._failure_tolerance() == 2

    def test_bo_phase_applies_budget_tolerance(self) -> None:
        # also covers a d_init the reference would not derive itself: n_splits
        # generalizes
        gen = BAxUSGenerator(
            vocs=_simple_vocs(n_vars=16),
            target_dim_init=4,
            n_initial_sobol=2,
            seed=0,
            eval_budget=210,
        )
        assert gen._failure_tolerance() == 4  # paper value; heuristic would say 2

        # ...and the fold-in honors it: one failure against a tolerance of 4 is
        # counted but must not shrink the region, which a hardwired tolerance of
        # 1 would
        row = {f"x{i}": 0.1 for i in range(16)}
        gen.add_data(pd.DataFrame([{**row, "f": 0.0}] * 3))  # 2 seeds + 1 BO row
        gen.trust_region.best_value = 1e9  # the BO result cannot improve on it
        gen._advance_trust_region()
        assert gen.trust_region.failure_counter == 1
        assert gen.trust_region.length == gen.length_init


# generator-level embedding expansion; the matrix-level split is
# TestBAxUSEmbedding's job. Right after an expansion the projected training data
# has duplicated coordinates, which can make the covariance indefinite: xopt
# downgrades the resulting fit failure to a warning and returns an untrained
# model, which the run must survive.
class TestBAxUSExpansion:
    def test_expansion_resets_the_region_and_survives_a_round_trip(self) -> None:
        # the generate that folds the results in expands first, then trains and
        # proposes in the new space, so its candidate doubles as the
        # post-expansion survival check. The tolerance is recomputed for the new
        # target_dim: d_init=2, n_splits=2, budget=200 -> round(4800/126)=38,
        # floor(38/6)=6. The expansion seed is derived from (seed, n_expansions),
        # so a run restored mid-flight expands to the same embedding.
        gen = BAxUSGenerator(
            vocs=_simple_vocs(n_vars=32), target_dim_init=2, seed=0, eval_budget=203
        )
        _seed_sobol(gen)
        restored = _round_trip(gen, with_data=True)

        row = {f"x{i}": 0.1 for i in range(32)}
        for generator in (gen, restored):
            _force_expansion(generator)  # best_value = 1e9 trips the restart
            generator.add_data(pd.DataFrame([{**row, "f": -1.0}]))
        candidate = gen.generate(1)[0]
        restored._advance_trust_region()  # the expansion, without the GP fit

        assert gen.embedding.target_dim == 8
        assert gen.trust_region.length == gen.length_init
        assert gen.trust_region.best_value == 1e9  # carried across the reset
        assert not gen.trust_region.restart_triggered
        assert gen.n_expansions == 1
        assert gen._failure_tolerance() == 6
        _assert_point_in_bounds(candidate, gen.vocs)
        assert all(math.isfinite(v) for v in candidate.values())
        assert restored.n_expansions == 1
        assert restored.embedding == gen.embedding

    def test_accumulated_failures_expand_organically(self) -> None:
        # the composition _force_expansion skips: real BO steps whose results
        # never improve, counted by generate's own advance until the region
        # collapses below length_min, restarts, and expands. Without a budget
        # the tolerance is ceil(2 / 2) = 1, so every miss halves; length_init
        # is shortened so the collapse takes 3 halvings instead of 7.
        gen = BAxUSGenerator(
            vocs=_simple_vocs(n_vars=16),
            target_dim_init=2,
            seed=0,
            length_init=0.5**4,
        )
        _seed_sobol(gen)

        ys = iter(range(0, -10, -1))  # the first fold-in succeeds, the rest fail
        for _ in range(10):
            point = gen.generate(1)[0]
            if gen.n_expansions:
                break
            gen.add_data(pd.DataFrame([{**point, "f": float(next(ys))}]))
        else:
            pytest.fail("failures never accumulated into an expansion")

        assert gen.embedding.target_dim == 8
        assert gen.trust_region.length == gen.length_init
        assert gen.trust_region.best_value == 0.0  # carried across the reset
        assert not gen.trust_region.restart_triggered
        assert gen._failure_tolerance() == 4  # recomputed: ceil(8 / 2)
        _assert_point_in_bounds(point, gen.vocs)
        assert all(math.isfinite(v) for v in point.values())


class TestBAxUSEmbedding:
    INPUT_DIM = 64
    B = 3

    @staticmethod
    def _bin_of(emb: BAxUSEmbedding, col: int) -> int:
        matrix = np.asarray(emb.matrix)
        return int(np.nonzero(matrix[:, col])[0][0])

    def _collision_rate(self, target_dim: int, trials: int = 200) -> float:
        # how often input dims 0 and 1 (adjacent in index) share a bin
        collisions = 0
        for trial in range(trials):
            emb = BAxUSEmbedding.create(
                input_dim=self.INPUT_DIM, target_dim=2, seed=trial
            )
            while emb.target_dim < target_dim:
                emb = emb.expand(self.B, seed=trial * 7919 + emb.target_dim)
            collisions += self._bin_of(emb, 0) == self._bin_of(emb, 1)
        return collisions / trials

    def test_create_structure_and_signs(self) -> None:
        # an all-positive matrix is structurally valid but biased, so the
        # structural check alone would not catch it
        emb = BAxUSEmbedding.create(input_dim=200, target_dim=4, seed=0)
        S = np.asarray(emb.matrix)
        assert S.shape == (4, 200)
        for col in range(200):
            assert np.count_nonzero(S[:, col]) == 1
            assert abs(S[:, col]).max() == 1.0

        positive = int((S == 1.0).sum())
        negative = int((S == -1.0).sum())
        assert positive and negative, "embedding signs are not randomized"
        assert 0.3 < positive / (positive + negative) < 0.7

    def test_lift_project_round_trip(self) -> None:
        # points lifted from target space project back exactly (S rows are orthogonal)
        emb = BAxUSEmbedding.create(input_dim=6, target_dim=2, seed=0)
        Z = np.array([[0.5, -0.25], [1.0, 1.0]])
        assert np.allclose(emb.project(emb.lift(Z)), Z)

    def test_expansion_separates_adjacent_dimensions(self) -> None:
        # a deterministic split leaves the collision rate pinned at
        # P(same initial bin) = 1/2, which structural checks cannot see: the
        # matrix is a valid partition either way
        rate_8 = self._collision_rate(target_dim=8)
        rate_32 = self._collision_rate(target_dim=32)

        assert rate_8 < 0.30, f"dims 0/1 still collide {rate_8:.0%} of the time at d=8"
        assert rate_32 < rate_8, (
            f"expanding did not decorrelate the split: {rate_8:.3f} -> {rate_32:.3f}"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_randomized_expansion_keeps_the_matrix_well_formed(self, seed: int) -> None:
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

        # at full dimensionality the column/row checks above force a signed
        # permutation by pigeonhole
        assert emb.target_dim == self.INPUT_DIM


# everything BAxUS refuses at construction. The constraints case guards a
# supports_constraints = False override against the True inherited from
# ExpectedImprovementGenerator, so it surfaces as VOCSError; everything else is
# BAxUS' own validator raising ValidationError. Discrete/contextual variables are
# base-class defaults BAxUS merely re-declares, covered by test_generator.py.
class TestBAxUSUnsupportedOptions:
    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            (
                {"fixed_features": {"x0": 0.5}},
                ValidationError,
                "support fixed_features",
            ),
            (
                {"max_travel_distances": [0.1] * 6},
                ValidationError,
                "max_travel_distances",
            ),
            ({"n_interpolate_points": 3}, ValidationError, "n_interpolate_points"),
            # custom_objective must clear its own CustomXoptObjective type validation
            # before ours can fire, so it needs a real (minimal) subclass instance
            (
                {"custom_objective": _MinimalCustomObjective(vocs=_simple_vocs())},
                ValidationError,
                "support custom_objective",
            ),
            ({"turbo_controller": "optimize"}, ValidationError, "no turbo controllers"),
            (
                _vocs_with(constraints={"c": ["LESS_THAN", 0.5]}),
                VOCSError,
                "constraints",
            ),
            (_vocs_with(observables=["g"]), ValidationError, "support observables"),
        ],
        ids=[
            "fixed_features",
            "max_travel_distances",
            "n_interpolate_points",
            "custom_objective",
            "turbo_controller",
            "constraints",
            "observables",
        ],
    )
    def test_rejected_at_construction(
        self, kwargs: dict[str, Any], error: type[Exception], match: str
    ) -> None:
        with pytest.raises(error, match=match):
            BAxUSGenerator(**{"vocs": _simple_vocs(), **kwargs})

    def test_declares_its_unsupported_features(self) -> None:
        gen = BAxUSGenerator(vocs=_simple_vocs())
        assert not gen.supports_batch_generation
        assert not gen.supports_constraints
        assert not gen.supports_discrete_variables
        assert not gen.supports_contextual_variables
        assert not gen.supports_no_objective


# saving and reloading must continue a run, not restart or advance it. By hand
# the resume contract is model_dump() then pop "model" - xopt ignores the exclude
# argument, so a trained generator's dump carries the live botorch model.
# Through Xopt.from_yaml the same set_data is what Xopt calls for a StateOwner,
# which must not replay the trust-region history the frame already reflects.
class TestBAxUSResume:
    def test_dump_is_yaml_safe_and_round_trips_components(self) -> None:
        # eval_budget rides along because the failure tolerance is derived from it
        # at every trust-region update: losing it silently downgrades a resumed
        # run to the ceil(target_dim / 2) heuristic.
        gen = BAxUSGenerator(vocs=_simple_vocs(), seed=7, eval_budget=50)
        _seed_sobol(gen)  # so sobol_draws is non-zero and has to survive the trip

        dump = gen.model_dump()
        dump.pop("model", None)

        mid = BAxUSGenerator(**dump)
        assert mid.embedding == gen.embedding
        assert mid.trust_region == gen.trust_region
        assert mid.eval_budget == gen.eval_budget
        assert mid.sobol_draws == gen.sobol_draws > 0

        mid_dump = mid.model_dump()
        mid_dump.pop("model", None)
        reloaded = yaml.safe_load(yaml.safe_dump(mid_dump))
        assert BAxUSGenerator(**reloaded).embedding == gen.embedding

        # Post-BO a live botorch model and a populated computation_time frame
        # ride along; without the pops, yaml.safe_dump raises RepresenterError
        # on the trained ModelListGP / the pd.DataFrame.
        point = gen.generate(1)[0]
        gen.add_data(pd.DataFrame([{**point, **_sphere(point)}]))
        bo_dump = gen.model_dump()
        bo_dump.pop("model", None)
        bo_dump.pop("computation_time", None)
        yaml.safe_dump(bo_dump)  # must not raise
        assert BAxUSGenerator(**bo_dump).trust_region == gen.trust_region

    # a restored run continues the sequence rather than restarting it: in the seed
    # phase bit-identically, in the BO phase only up to GP fitting and
    # acquisition-optimization noise, so both are driven from the same torch seed
    @pytest.mark.parametrize("phase", ["sobol", "bo"])
    def test_round_trip_proposes_the_same_next_point(self, phase: str) -> None:
        g1 = BAxUSGenerator(vocs=_simple_vocs(n_vars=4), target_dim_init=2, seed=7)
        if phase == "sobol":
            # stop one point short of the quota, so the next call still seeds
            for _ in range(g1.n_initial_sobol - 1):
                point = g1.generate(1)[0]
                g1.add_data(pd.DataFrame([{**point, **_sphere(point)}]))
        else:
            _seed_sobol(g1)
            # a post-quota row by hand, so the round trip starts in the BO phase
            # without paying for a fit here; the compared generates fit anyway
            g1.add_data(pd.DataFrame([{**{f"x{i}": 0.1 for i in range(4)}, "f": -0.5}]))

        g2 = _round_trip(g1, with_data=True)

        torch.manual_seed(0)
        p1 = g1.generate(1)[0]
        torch.manual_seed(0)
        p2 = g2.generate(1)[0]
        assert p1 == pytest.approx(p2)

    @staticmethod
    def _state(gen: BAxUSGenerator) -> dict[str, Any]:
        tr = gen.trust_region
        return {
            "target_dim": gen.embedding.target_dim,
            "matrix": gen.embedding.matrix,
            "length": tr.length,
            "success_counter": tr.success_counter,
            "failure_counter": tr.failure_counter,
            "failure_tolerance": gen._failure_tolerance(),
            "best_value": tr.best_value,
            "sobol_draws": gen.sobol_draws,
            "n_expansions": gen.n_expansions,
            "tr_observed_rows": gen.tr_observed_rows,
        }

    def _run(self, steps: int = 6) -> Xopt:
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

    def test_checkpoint_cycles_hold_state_and_resume_folds_in_only_new_results(
        self,
    ) -> None:
        # no evaluations happen between cycles, so nothing may change across them.
        # The proposed point is not compared exactly: xopt's yaml writer keeps 10
        # significant digits, and that ~1e-10 perturbation is amplified by GP
        # hyperparameter fitting and the L-BFGS acquisition optimization.
        X = self._run()
        before = self._state(X.generator)
        assert len(X.data) == 6
        # the row the resumed run must fold in: the last one evaluated pre-restore
        new_y = float(X.data["f"].to_numpy(dtype=np.float64)[-1])
        oracle = X.generator.trust_region.model_copy(deep=True)

        resumed = X
        for cycle in range(2):
            resumed = Xopt.from_yaml(resumed.yaml())
            assert len(resumed.data) == 6, f"data changed on cycle {cycle}"
            assert self._state(resumed.generator) == before, (
                f"generator state drifted on checkpoint cycle {cycle}"
            )

        for runner in (X, resumed):
            runner.step()
        after = self._state(resumed.generator)

        assert after["tr_observed_rows"] == before["tr_observed_rows"] + 1
        assert after["n_expansions"] == before["n_expansions"]

        # exactly one tick, not "at most one": replaying that single observation
        # through a copy of the pre-restore region must reproduce the state. A
        # replay of the history would move the counters further (or not at all).
        oracle.update(
            torch.tensor([X.generator._objective_weight() * new_y], dtype=torch.double),
            X.generator._failure_tolerance(),
        )
        assert (after["success_counter"], after["failure_counter"]) == (
            oracle.success_counter,
            oracle.failure_counter,
        )
        assert after["length"] == pytest.approx(oracle.length, rel=1e-12)

        # ...and the same step on the un-checkpointed run lands in the same place;
        # only the components the oracle replay above does not already pin
        original, restored = self._state(X.generator), self._state(resumed.generator)
        for key in ("target_dim", "matrix", "failure_tolerance", "sobol_draws"):
            assert original[key] == restored[key], f"{key} diverged after resume"
        assert original["best_value"] == pytest.approx(restored["best_value"], rel=1e-6)


# the trust-region box, and the lengthscale weights that shape it. No end-to-end
# test can pin the box down: replacing _get_optimization_bounds with the full
# domain, or centering it on the worst point, leaves every behavioral test
# passing - it needs direct assertions on the returned box.
class TestBAxUSOptimizationBounds:
    Z_ROWS = [[0.1, 0.7], [-0.6, 0.3], [0.5, -0.4], [-0.2, -0.8]]
    YS = [-1.0, -2.0, 5.0, -3.0]  # row 2 best; row 3 is what argmin would pick
    BEST_ROW = 2

    @classmethod
    def _gen(cls, direction: str = "MAXIMIZE") -> BAxUSGenerator:
        return _gen_with_target_space_data(cls.Z_ROWS, cls.YS, direction=direction)

    # argmin instead of argmax here would center on the worst point, and the
    # incumbent is direction-aware rather than just the numeric maximum. In
    # [-1, 1] target units the unclamped width is 2 * length (the length fields
    # are in [0, 1] units, per the BoTorch reference).
    @pytest.mark.parametrize(
        ("direction", "length"),
        [("MAXIMIZE", 0.1), ("MAXIMIZE", 0.25), ("MAXIMIZE", 0.4), ("MINIMIZE", 0.4)],
    )
    def test_box_is_centered_on_the_incumbent_and_2_length_wide(
        self, direction: str, length: float
    ) -> None:
        gen = self._gen(direction)
        gen.trust_region.length = length

        bounds = gen._get_optimization_bounds()
        center = (bounds[0] + bounds[1]) / 2.0
        width = bounds[1] - bounds[0]

        incumbent = _in_target_space(gen, gen.data)[self.BEST_ROW]
        assert torch.allclose(
            center, torch.tensor(incumbent, dtype=torch.double), atol=1e-12
        )
        assert torch.allclose(
            width, torch.full_like(width, 2.0 * length), atol=1e-12
        ), f"expected width {2.0 * length}, got {width.tolist()}"

    def test_box_stays_inside_the_domain(self) -> None:
        # at length_max the box is wider than the domain on both sides of the
        # incumbent, so it must clamp rather than overflow
        gen = self._gen()
        gen.trust_region.length = gen.trust_region.length_max

        bounds = gen._get_optimization_bounds()

        assert bool((bounds[0] >= -1.0).all())
        assert bool((bounds[1] <= 1.0).all())
        assert bool((bounds[1] > bounds[0]).all())

    def test_lengthscale_weights_stretch_the_box_per_dimension(self) -> None:
        gen = self._gen()
        gen.trust_region.length = 0.2
        gen._lengthscale_weights = lambda model: torch.tensor(  # type: ignore[method-assign]
            [0.5, 2.0], dtype=torch.double
        )

        weighted = gen._get_optimization_bounds()

        # 2 * length * weight per dimension, unclamped at this incumbent
        width = weighted[1] - weighted[0]
        assert torch.allclose(
            width, torch.tensor([0.2, 0.8], dtype=torch.double), atol=1e-12
        ), f"expected widths [0.2, 0.8], got {width.tolist()}"

    @staticmethod
    def _kernel_model(lengthscale: torch.Tensor) -> MagicMock:
        # a mock shaped like a gpytorch model: covar_module.base_kernel.lengthscale
        kernel = MagicMock()
        kernel.lengthscale = lengthscale
        covar = MagicMock()
        covar.base_kernel = kernel
        model = MagicMock(spec=["covar_module"])
        model.covar_module = covar
        return model

    def test_kernel_lengthscale_is_used_with_a_uniform_fallback(self) -> None:
        # a vector lengthscale gives non-uniform weights, a scalar one is
        # unsqueezed rather than collapsing to a 0-dim tensor, and a model with no
        # kernel at all falls back to an isotropic region
        gen = _bare_gen()
        model = self._kernel_model(torch.tensor([[0.5, 2.0]], dtype=torch.double))

        weights = gen._lengthscale_weights(model)
        assert weights.shape == (2,)
        assert not torch.allclose(weights, torch.ones(2, dtype=torch.double))

        scalar_gen = BAxUSGenerator(vocs=_simple_vocs(n_vars=4), target_dim_init=1)
        scalar_model = self._kernel_model(torch.tensor(0.5, dtype=torch.double))
        assert scalar_gen._lengthscale_weights(scalar_model).shape == (1,)

        bare = MagicMock(spec=["posterior", "num_outputs"])
        assert torch.allclose(
            gen._lengthscale_weights(bare), torch.ones(2, dtype=torch.double)
        )

    # forming the raw product before taking the root underflows to zero once
    # target_dim reaches a few hundred, making every weight inf and silently
    # widening the trust region to the entire domain
    @pytest.mark.parametrize("target_dim", [8, 1200])
    def test_weights_are_finite_volume_preserving_and_ordered(
        self, target_dim: int
    ) -> None:
        torch.manual_seed(0)
        lengthscales = (
            torch.distributions.LogNormal(0.0, 2.0).sample((target_dim,)).double()
        )

        weights = _normalize_lengthscale_weights(lengthscales, target_dim)

        assert bool(torch.isfinite(weights).all()), "weights underflowed to inf"
        assert bool((weights > 0).all())
        # unit geometric mean
        assert float(weights.log().mean().exp()) == pytest.approx(1.0, rel=1e-9)
        # the ordering keeps the normalization from inverting the weights
        assert weights.argsort().tolist() == lengthscales.argsort().tolist()


# analytic LogEI on the target-space model with best_f = the best weighted finite
# objective, inherited from ExpectedImprovementGenerator. No objective-value
# comparison can establish that the candidate comes from it: on a quadratic the
# center of a symmetric domain beats the average random point for *any* placement
# of the optimum, so a generator that always proposed the center would still
# "beat Sobol on average" - the candidate has to be checked against the box.
class TestBAxUSAcquisition:
    @pytest.mark.parametrize("direction", ["MAXIMIZE", "MINIMIZE"])
    def test_best_f_is_the_best_weighted_finite_objective(self, direction: str) -> None:
        # MINIMIZE negates; a NaN row never reaches best_f in either direction
        vocs = VOCS(
            variables={f"x{i}": [-1.0, 1.0] for i in range(4)},
            objectives={"f": direction},
        )
        gen = BAxUSGenerator(vocs=vocs, target_dim_init=2, seed=0)
        gen.add_data(
            pd.DataFrame(
                [
                    {**{f"x{i}": x for i in range(4)}, "f": f}
                    for x, f in [
                        (-0.5, -3.0),
                        (0.1, 2.0),
                        (0.6, 0.5),
                        (0.9, float("nan")),
                    ]
                ]
            )
        )

        acq = gen.get_acquisition(gen.train_model())

        ys = gen.data["f"].to_numpy(dtype=np.float64)
        expected = np.nanmax(ys) if direction == "MAXIMIZE" else -np.nanmin(ys)
        assert isinstance(acq, LogExpectedImprovement)
        assert float(cast(torch.Tensor, acq.best_f)) == pytest.approx(expected)

    # five clearly worse points to surround the incumbent with
    Z_OTHERS = [[-0.9, 0.9], [0.2, 0.4], [-0.3, -0.6], [0.6, 0.1], [-0.5, -0.2]]

    @classmethod
    def _gen_with_incumbent_at(cls, z_star: list[float]) -> BAxUSGenerator:
        return _gen_with_target_space_data(
            [z_star, *cls.Z_OTHERS], [10.0, *[-5.0] * 5], n_vars=6
        )

    def test_candidate_stays_in_the_box_and_tracks_the_incumbent(self) -> None:
        # lengthscale weights are pinned to 1 so the box is exactly ``length`` wide
        # around the incumbent; left to a real fit, a handful of points produces
        # extreme ARD ratios that widen one side to the whole domain
        points = []
        for z_star in ([0.7, 0.7], [-0.7, -0.7]):
            gen = self._gen_with_incumbent_at(z_star)

            gen._advance_trust_region()  # fold the results in without a fit
            gen.trust_region.length = 0.2
            gen._lengthscale_weights = lambda model: torch.ones(  # type: ignore[method-assign]
                2, dtype=torch.double
            )
            candidate = gen.generate(1)[0]
            z = _in_target_space(gen, pd.DataFrame([candidate]))[0]

            # no data was added, so this reproduces the box the candidate came from
            bounds = gen._get_optimization_bounds()
            lb, ub = bounds[0].numpy(), bounds[1].numpy()
            assert not ((lb <= 0.0).all() and (ub >= 0.0).all()), (
                "box contains the origin: cannot detect a constant proposal"
            )
            assert (z >= lb - 1e-9).all(), f"candidate {z} below trust region {lb}"
            assert (z <= ub + 1e-9).all(), f"candidate {z} above trust region {ub}"
            points.append(z)

        assert points[0][0] > points[1][0], (
            f"proposal did not follow the incumbent: {points[0]} vs {points[1]}"
        )
        assert points[0][1] > points[1][1]
