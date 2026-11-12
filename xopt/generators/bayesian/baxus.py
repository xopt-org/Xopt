"""BAxUS generator - Bayesian Optimization with Adaptively Expanding Subspaces.

Optimizes in a low-dimensional random embedding of the input space and expands it
whenever the trust region shrinks below a threshold, which makes it effective when
many input dimensions are irrelevant.

The analytic LogEI acquisition is inherited from ``ExpectedImprovementGenerator``
and re-based into the embedded target space through ``get_training_data``,
``train_model``, ``_get_torch_bounds`` / ``_get_optimization_bounds`` and
``_process_candidates``. All persistent state lives in serializable pydantic
fields and the generator is a ``StateOwner``, so a dump -> construct round trip
reproduces the run.

Deviations from the reference: ``target_dim_init`` is a user knob rather than
derived from the input dimensionality, the acquisition is the inherited analytic
LogEI over the trust-region box rather than Thompson sampling over a masked
candidate set, and data not generated through the current embedding is
least-squares projected into target space.

Papenmeier et al., "Increasing the Scope as You Learn: Adaptive Bayesian
Optimization in Nested Subspaces", NeurIPS 2022
https://arxiv.org/abs/2304.11468
Tutorial: https://botorch.org/docs/tutorials/baxus
"""

import logging
import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
from pydantic import (
    Field,
    PositiveInt,
    PrivateAttr,
    SerializeAsAny,
    model_validator,
)
from torch.nn import Module
from torch.quasirandom import SobolEngine

from xopt.errors import GeneratorWarning
from xopt.generator import StateOwner
from xopt.generators.bayesian.base_model import ModelConstructor
from xopt.generators.bayesian.bayesian_generator import formatted_base_docstring
from xopt.generators.bayesian.expected_improvement import ExpectedImprovementGenerator
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.bayesian.utils import set_botorch_weights
from xopt.pydantic import XoptBaseModel
from xopt.vocs import VOCS, convert_numpy_to_inputs, get_variable_bounds_array

logger = logging.getLogger(__name__)


def _normalize_lengthscale_weights(
    lengthscales: torch.Tensor, target_dim: int
) -> torch.Tensor:
    """Scale lengthscales to mean 1, then to unit geometric mean.

    The geometric mean is taken as a product of per-element roots (the reference
    form); forming the product first underflows to zero past a few hundred
    dimensions, making every weight inf.
    """
    weights = lengthscales / lengthscales.mean()
    return weights / weights.pow(1.0 / target_dim).prod()


class BAxUSTrustRegion(XoptBaseModel):
    """Trust-region state, in normalized [0, 1] target-space units.

    The generator owns dimensionality (``embedding.target_dim``) and passes its
    derived failure tolerance into every ``update``.
    """

    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    success_counter: int = 0
    success_tolerance: int = 3
    # None until a BO-phase evaluation is folded in
    best_value: float | None = None
    restart_triggered: bool = False

    def update(self, y_weighted: torch.Tensor, failure_tolerance: int) -> None:
        """Adjust the trust region from weighted objective values (higher is better)."""
        best = self.best_value
        y_max = float(y_weighted.max())
        improved = best is None or y_max > best + 1e-3 * abs(best)
        if improved:
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter >= self.success_tolerance:
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter >= failure_tolerance:
            self.length /= 2.0
            self.failure_counter = 0

        self.best_value = y_max if best is None else max(best, y_max)

        if self.length < self.length_min:
            self.restart_triggered = True


class BAxUSEmbedding(XoptBaseModel):
    """Sparse random embedding of shape (target_dim, input_dim).

    Exactly one signed unit entry per column. ``create`` and ``expand`` both
    consume randomness and take an explicit seed, so a run can be reproduced from
    a dump.
    """

    matrix: list[list[float]]

    @classmethod
    def create(
        cls, input_dim: int, target_dim: int, seed: int | None
    ) -> "BAxUSEmbedding":
        rng = np.random.default_rng(seed)
        S = np.zeros((target_dim, input_dim), dtype=np.float64)
        perm = rng.permutation(input_dim)
        target_rows = np.arange(input_dim) % target_dim
        signs = rng.choice([-1.0, 1.0], size=input_dim)
        S[target_rows, perm] = signs
        return cls(matrix=S.tolist())

    @property
    def target_dim(self) -> int:
        return len(self.matrix)

    @property
    def input_dim(self) -> int:
        return len(self.matrix[0])

    def lift(self, Z: np.ndarray) -> np.ndarray:
        """Lift from target subspace to full input space: ``X = Z @ S``."""
        return Z @ np.asarray(self.matrix, dtype=np.float64)

    def project(self, X: np.ndarray) -> np.ndarray:
        """Project from full input space to target subspace via the pseudo-inverse."""
        return X @ np.linalg.pinv(np.asarray(self.matrix, dtype=np.float64))

    def expand(
        self, new_bins_on_split: int, seed: int | None = None
    ) -> "BAxUSEmbedding":
        """Split each bin into up to ``new_bins_on_split + 1`` sub-bins, keeping signs.

        Contributing dimensions are permuted before the split, matching the
        reference - otherwise the split is a fixed function of column index, so
        dimensions adjacent in index that share a bin are never separated before
        full dimensionality.
        """
        rng = np.random.default_rng(seed)
        S = np.asarray(self.matrix, dtype=np.float64)
        old_target_dim, input_dim = S.shape
        n_sub = new_bins_on_split + 1
        new_target_dim = min(old_target_dim * n_sub, input_dim)

        new_S = np.zeros((new_target_dim, input_dim), dtype=np.float64)
        new_row = 0
        for row in S:
            bin_cols = rng.permutation(np.nonzero(row)[0])
            for sub_cols in np.array_split(bin_cols, min(n_sub, bin_cols.size)):
                new_S[new_row, sub_cols] = row[sub_cols]
                new_row += 1
        return BAxUSEmbedding(matrix=new_S.tolist())


class BAxUSGenerator(ExpectedImprovementGenerator, StateOwner):
    name = "baxus"
    supports_batch_generation: bool = False
    # the embedded target-space GP models only the objective over continuous
    # variables; constrained, discrete, observable, and contextual vocs are rejected
    supports_constraints: bool = False
    supports_discrete_variables: bool = False
    supports_contextual_variables: bool = False
    supports_no_objective: bool = False

    __doc__ = (
        "Bayesian optimization with adaptively expanding subspaces (BAxUS)\n"
        + formatted_base_docstring()
    )

    # BAxUS manages its own trust region in the embedded target space
    _compatible_turbo_controllers = []

    target_dim_init: PositiveInt = Field(
        default=2,
        description="initial target dimensionality of the embedding",
    )
    n_initial_sobol: int = Field(
        default=0,
        ge=0,
        description="number of initial Sobol points (0 = auto-computed)",
    )
    length_init: float = Field(
        default=0.8,
        description="initial trust-region side length",
    )
    new_bins_on_split: PositiveInt = Field(
        default=3,
        description="number of new bins created per existing bin on expansion",
    )
    seed: int | None = Field(
        default=None,
        description=(
            "random seed for the embedding and Sobol initialization; GP fitting and "
            "acquisition optimization use torch's global RNG instead"
        ),
    )
    eval_budget: PositiveInt | None = Field(
        default=None,
        description=(
            "total planned evaluations for the run, seed points included; enables the "
            "reference budget-aware failure tolerance, None keeps ceil(target_dim / 2)"
        ),
    )
    embedding: BAxUSEmbedding = Field(
        description="sparse random embedding (auto-created from vocs and seed when omitted)",
    )
    trust_region: BAxUSTrustRegion = Field(
        description="trust-region state (auto-created when omitted)",
    )
    sobol_draws: int = Field(
        default=0,
        ge=0,
        description="number of Sobol seed points drawn so far, used to resume the sequence",
    )
    n_expansions: int = Field(
        default=0,
        ge=0,
        description=(
            "number of embedding expansions so far, combined with seed to keep each "
            "expansion reproducible across a round trip"
        ),
    )
    tr_observed_rows: int = Field(
        default=0,
        ge=0,
        description=(
            "number of finite-objective rows already folded into the trust region, "
            "which prevents re-ingesting history on resume"
        ),
    )
    # plain default instance, matching BayesianGenerator: pydantic deep-copies it
    # per instance, and get_generator_defaults only understands `default`
    gp_constructor: SerializeAsAny[ModelConstructor] = Field(
        StandardModelConstructor(use_low_noise_prior=True),
        description=(
            "constructor used to generate the target-space model; the low-noise prior "
            "matches the near-interpolating GP the reference trust-region logic assumes"
        ),
    )

    _sobol: SobolEngine | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _init_components(cls, values: Any) -> Any:
        """Create the embedding and trust region, and size the Sobol seed quota.

        A dump -> construct round trip supplies these and skips their creation.
        """
        if not isinstance(values, dict) or values.get("vocs") is None:
            return values
        vocs = values["vocs"]
        if isinstance(vocs, dict):
            # xopt's VOCS validators pop "type" off entry dicts in place, so write
            # the parsed object back before the outer "vocs" field validation
            values["vocs"] = vocs = VOCS(**vocs)
        input_dim = len(vocs.variable_names)

        # this runs before pydantic applies defaults, so the `or` fallbacks below
        # must mirror the target_dim_init / length_init field defaults
        emb = values.get("embedding")
        if emb is None:
            target_dim = min(int(values.get("target_dim_init") or 2), input_dim)
            emb = BAxUSEmbedding.create(
                input_dim=input_dim, target_dim=target_dim, seed=values.get("seed")
            )
            values["embedding"] = emb
        # the embedding arrives as a raw dict on a dump -> construct round trip
        emb_target_dim = len(emb["matrix"]) if isinstance(emb, dict) else emb.target_dim

        if not values.get("n_initial_sobol"):
            values["n_initial_sobol"] = max(2, emb_target_dim + 1)

        if values.get("trust_region") is None:
            values["trust_region"] = BAxUSTrustRegion(
                length=float(values.get("length_init") or 0.8)
            )

        return values

    def model_post_init(self, context: Any, /) -> None:
        """Warn once if the budget is missing."""
        # runs exactly once per construction; a model validator of either mode
        # would re-fire on every field assignment under validate_assignment
        super().model_post_init(context)
        if self.eval_budget is None:
            warnings.warn(
                "BAxUS: eval_budget is not set, so the reference budget-aware failure "
                "tolerance is replaced by the ceil(target_dim / 2) heuristic. The "
                "embedding then expands far more slowly and may never reach full "
                "dimensionality within a typical run - set eval_budget to the total "
                "number of planned evaluations.",
                GeneratorWarning,
                stacklevel=2,
            )

    @model_validator(mode="after")
    def _validate_budget_covers_seeding(self) -> "BAxUSGenerator":
        """Reject an eval_budget smaller than the Sobol seed quota."""
        if self.eval_budget is not None and self.eval_budget < self.n_initial_sobol:
            raise ValueError(
                f"eval_budget ({self.eval_budget}) must cover the seed quota (n_initial_sobol={self.n_initial_sobol})"
            )
        return self

    @model_validator(mode="after")
    def _reject_unsupported_options(self) -> "BAxUSGenerator":
        """vocs-space point controls cannot be honored in an embedded subspace."""
        for option in (
            "fixed_features",
            "max_travel_distances",
            "custom_objective",
            "n_interpolate_points",
        ):
            if getattr(self, option) is not None:
                raise ValueError(f"BAxUS does not support {option}")
        # the target-space model has a single outcome, while the inherited
        # acquisition scalarizes over every vocs output
        if self.vocs.observables:
            raise ValueError("BAxUS does not support observables")
        return self

    def generate(self, n_candidates: int = 1) -> list[dict[str, float]]:
        """Sobol seed points first, then LogEI within the trust region."""
        # repeated from the base call, which the seed branch below never reaches
        if n_candidates > 1 and not self.supports_batch_generation:
            raise NotImplementedError(
                "This Bayesian algorithm does not currently support parallel candidate generation"
            )

        if len(self._finite_data()) < self.n_initial_sobol:
            z = torch.tensor(self._draw_sobol_point())
            return self._process_candidates(z).to_dict("records")

        # fold newly observed results in before training, so an expansion
        # rebuilds the model in the new target space
        self._advance_trust_region()
        return super().generate(n_candidates)

    def get_training_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Restrict the GP training set to rows with a finite objective."""
        return data[self._finite_objective_mask(data)]

    def _process_candidates(self, candidates: torch.Tensor) -> pd.DataFrame:
        """Lift target-space candidates into vocs space.

        Replaces the base implementation, whose discrete snapping and
        fixed-feature handling are keyed to vocs-space columns.
        """
        x_raw = self._lift_to_vocs(candidates.detach().cpu().numpy())
        return convert_numpy_to_inputs(self.vocs, x_raw, include_constants=False)

    def _get_torch_bounds(self) -> torch.Tensor:
        """The full embedded target space, ``[-1, 1]^target_dim``.

        Re-points a base-class seam from vocs space into target space; safe
        because every other consumer is overridden or rejected.
        """
        ones = torch.ones(self.embedding.target_dim, dtype=torch.double)
        return torch.stack([-ones, ones])

    def get_optimum(self) -> pd.DataFrame:
        """Best point given by the model's posterior mean, lifted to vocs space."""
        # an expansion clears the model, and a freshly restored run has none
        if self.model is None:
            self.train_model()
        return super().get_optimum()

    def visualize_model(self, **kwargs):
        """Not supported: the model lives in the embedded target space."""
        raise NotImplementedError(
            "BAxUS models live in the embedded target space; vocs-space model "
            "visualization is not supported"
        )

    def add_data(self, new_data: pd.DataFrame) -> None:
        """Ingest evaluation results.

        Pure ingestion - the trust region advances in ``generate`` instead, so the
        state machine follows optimization iterations rather than ``add_data``
        chunking. Rows without a finite objective stay in ``data`` but are kept
        out of the GP training set and the trust region.
        """
        super().add_data(new_data)

        n_bad = int((~self._finite_objective_mask(new_data)).sum())
        if n_bad:
            logger.warning(
                "BAxUS: dropping %d row(s) with non-finite objective from training data",
                n_bad,
            )

    def set_data(self, data: pd.DataFrame) -> None:
        """Reattach a full dataset without replaying trust-region updates.

        ``Xopt`` calls this instead of ``add_data`` when loading a saved run (see
        ``StateOwner``).
        """
        self.data = data

    def _advance_trust_region(self) -> None:
        """Fold newly observed BO-phase results into the trust region.

        Called once per ``generate``, mirroring how ``BayesianGenerator`` drives
        ``TurboController.update_state``. Sobol seed points never move the region
        (matching the reference), and ``tr_observed_rows`` keeps a resumed run
        from re-ingesting history.
        """
        data = self._finite_data()
        n_finite = len(data)
        start = max(self.tr_observed_rows, self.n_initial_sobol)
        if n_finite <= start:
            return

        y = data[self._objective_name].to_numpy(dtype=np.float64)[start:]
        self.tr_observed_rows = n_finite

        self.trust_region.update(
            torch.tensor(self._objective_weight() * y, dtype=torch.double),
            self._failure_tolerance(),
        )

        if self.trust_region.restart_triggered:
            previous_dim = self.embedding.target_dim
            self.embedding = self.embedding.expand(
                self.new_bins_on_split,
                # derived seed, so a round trip reproduces the expansion
                seed=None
                if self.seed is None
                else self.seed + 104729 * (self.n_expansions + 1),
            )
            logger.info(
                "BAxUS: trust region too small (%.4e), expanded embedding from %d to %d dims",
                self.trust_region.length,
                previous_dim,
                self.embedding.target_dim,
            )
            self.n_expansions += 1
            # the reference reset, minus its counter zeroing: the shrink that
            # trips a restart has just reset both counters, so only length and
            # the flag can differ - and best_value deliberately survives
            self.trust_region.length = self.length_init
            self.trust_region.restart_triggered = False
            # the fitted model lives in the old target space
            self.model = None

    def train_model(
        self, data: pd.DataFrame | None = None, update_internal: bool = True
    ) -> Module:
        """Fit the GP on target-space projections of the finite-objective data."""
        data = data if data is not None else self._finite_data()
        if data.empty:
            raise ValueError("no data available to build model")

        Z = self.embedding.project(self._normalized_inputs(data))
        input_names = [f"z{i}" for i in range(self.embedding.target_dim)]
        objective_name = self._objective_name
        frame = pd.DataFrame(Z, columns=input_names)
        frame[objective_name] = data[objective_name].to_numpy(dtype=np.float64)

        # xopt's constructor downgrades a failed fit to a warning and returns an
        # untrained model, which is survivable and happens occasionally right
        # after an embedding expansion
        model = self.gp_constructor.build_model(
            input_names=input_names,
            outcome_names=[objective_name],
            data=frame,
            input_bounds={name: [-1.0, 1.0] for name in input_names},
            **self.tkwargs,
        )
        if update_internal:
            self.model = model
        return model

    def _get_optimization_bounds(self) -> torch.Tensor:
        """Trust-region bounds in target space, ``[-1, 1]^target_dim``.

        The reference ``create_candidate`` box: centered on the incumbent with
        half-width ``length * weights``, clamped to the domain. A half-width of
        ``length`` in this side-2 domain covers the same fraction of each side as
        TuRBO's ``length / 2`` does in its unit domain.
        """
        data = self._finite_data()
        y = torch.tensor(
            data[self._objective_name].to_numpy(dtype=np.float64),
            dtype=torch.double,
        )
        best_idx = int(torch.argmax(self._objective_weight() * y))

        Z = self.embedding.project(self._normalized_inputs(data))
        center = torch.tensor(Z[best_idx], dtype=torch.double)

        half_width = self.trust_region.length * self._lengthscale_weights(self.model)
        lb = torch.clamp(center - half_width, -1.0, 1.0)
        ub = torch.clamp(center + half_width, -1.0, 1.0)
        return torch.stack([lb, ub]).to(**self.tkwargs)

    def _failure_tolerance(self) -> int:
        """The trust region's failure tolerance, budget-aware when possible.

        Derived on demand, as in the reference, so a changed ``eval_budget`` or an
        expanded embedding is honored automatically. The reference schedule
        (Papenmeier et al., Alg. 1) paces expansions so the final split lands as
        the budget runs out; without ``eval_budget`` it falls back to
        ``ceil(target_dim / 2)``, and at full dimensionality it widens to
        ``target_dim``. The planned split count is generalized to
        ``ceil(log_{b+1}(input_dim / d_init))`` to honor ``target_dim_init``.
        """
        target_dim = self.embedding.target_dim
        input_dim = self.embedding.input_dim
        if target_dim >= input_dim:
            return target_dim
        if self.eval_budget is None:
            return math.ceil(target_dim / 2)
        d_init = min(self.target_dim_init, input_dim)
        b = self.new_bins_on_split
        # the epsilon keeps exact powers of b + 1 from rounding up on float noise
        n_splits = math.ceil(math.log(input_dim / d_init, b + 1) - 1e-9)
        # each split multiplies dimensionality by (b + 1)
        growth = (b + 1) ** (n_splits + 1)
        # evaluations left for the BO phase, then this target_dim's share of them
        bo_budget = max(1, self.eval_budget - self.n_initial_sobol)
        split_budget = round(b * bo_budget * target_dim / (d_init * (growth - 1)))
        # max(1, ...): length_init == length_min would otherwise divide by zero
        halvings = max(
            1,
            math.floor(math.log(self.trust_region.length_min / self.length_init, 0.5)),
        )
        # spend that share evenly over the halvings needed to reach length_min
        return min(target_dim, max(1, math.floor(split_budget / halvings)))

    @property
    def _objective_name(self) -> str:
        """The single objective this generator models (multi-objective is rejected)."""
        return self.vocs.objective_names[0]

    def _lift_to_vocs(self, z: np.ndarray) -> np.ndarray:
        """Lift through the embedding and scale to vocs bounds.

        No clamp is needed: every caller optimizes or draws within [-1, 1], and the
        embedding maps each coordinate to a signed copy of one input.
        """
        x_norm = self.embedding.lift(z)
        lb, ub = get_variable_bounds_array(self.vocs)
        return lb + (x_norm + 1.0) / 2.0 * (ub - lb)

    def _objective_weight(self) -> float:
        """xopt objective weight: +1 for MAXIMIZE, -1 for MINIMIZE."""
        weights = set_botorch_weights(self.vocs)
        return float(weights[self.vocs.output_names.index(self._objective_name)])

    def _finite_objective_mask(self, frame: pd.DataFrame) -> np.ndarray:
        """Rows of ``frame`` with a finite objective; all-False if the column is absent.

        ``vocs.extract_data(return_valid=True)`` cannot stand in - it filters on
        feasibility, which is all-True without constraints.
        """
        column = frame.get(self._objective_name)
        if column is None:
            # every evaluation in the frame failed, so the column was never created
            return np.zeros(len(frame), dtype=bool)
        y = pd.to_numeric(column, errors="coerce").to_numpy(dtype=np.float64)
        return np.isfinite(y)

    def _finite_data(self) -> pd.DataFrame:
        """Rows of ``data`` with a finite objective - the GP training set."""
        if self.data is None:
            return pd.DataFrame()
        return self.get_training_data(self.data)

    def _normalized_inputs(self, data: pd.DataFrame) -> np.ndarray:
        """Variable columns scaled to [-1, 1] per vocs bounds.

        [-1, 1] rather than ``vocs.normalize_inputs``' [0, 1]: these feed a matrix
        product with the embedding, which is sign-symmetric.
        """
        lb, ub = get_variable_bounds_array(self.vocs)
        X = data[self.vocs.variable_names].to_numpy(dtype=np.float64)
        return 2.0 * (X - lb) / (ub - lb) - 1.0

    def _draw_sobol_point(self) -> np.ndarray:
        """Next quasi-random seed point in target space, ``[-1, 1]^target_dim``.

        The cached engine is rebuilt and fast-forwarded by ``sobol_draws`` after a
        round trip (exact continuation needs a fixed ``seed``), and whenever the
        embedding has expanded under it - which ``set_data`` can expose by dropping
        back below the seed quota.
        """
        target_dim = self.embedding.target_dim
        if self._sobol is None or self._sobol.dimension != target_dim:
            self._sobol = SobolEngine(
                dimension=target_dim, scramble=True, seed=self.seed
            )
            if self.sobol_draws:
                self._sobol.fast_forward(self.sobol_draws)
        z = 2.0 * self._sobol.draw(1).to(dtype=torch.double).numpy() - 1.0
        self.sobol_draws += 1
        return z

    def _lengthscale_weights(self, model: Module) -> torch.Tensor:
        """Per-dimension trust-region scaling weights from the fitted model.

        Falls back to uniform weights (an isotropic region) for kernels that have
        no lengthscale.
        """
        target_dim = self.embedding.target_dim
        single = model.models[0] if hasattr(model, "models") else model

        # each getattr tolerates a None input, so the chain needs no nesting
        covar = getattr(single, "covar_module", None)
        kernel = getattr(covar, "base_kernel", covar)
        lengthscale = getattr(kernel, "lengthscale", None)
        if lengthscale is not None:
            ls = torch.atleast_1d(lengthscale.detach().squeeze())
            return _normalize_lengthscale_weights(ls, target_dim)

        return torch.ones(target_dim, dtype=torch.double)
