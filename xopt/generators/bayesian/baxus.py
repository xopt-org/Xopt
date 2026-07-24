"""BAxUS generator - Bayesian Optimization with Adaptively Expanding Subspaces.

Starts in a low-dimensional random subspace and gradually expands it, making
it effective when many input dimensions are irrelevant.

The inherited ``data`` frame is the single source of truth; all persistent
state lives in serializable pydantic fields, and the generator is a
``StateOwner`` so ``Xopt.from_yaml`` reattaches data without replaying the
trust-region history it already reflects. The trust region advances once per
``generate`` call (as ``TurboController`` does), which keeps the trajectory
independent of how results were batched into ``add_data``.

Randomness is seeded rather than removed: the embedding's creation and each of
its expansions draw from ``seed`` combined with the expansion count, so a
dump -> construct round trip reproduces the same subspaces.

The BO phase is ``BayesianGenerator.generate`` re-based into the embedded target
space through four override seams - ``get_training_data`` (finite-objective rows
only), ``train_model`` (fit on projections), ``_get_torch_bounds`` /
``_get_optimization_bounds`` (target-space boxes), and ``_process_candidates``
(lift back to vocs space). Everything else in that pipeline is inert here,
because the options it serves are rejected at construction.

Deviations from the reference:
    - ``target_dim_init`` is a user knob rather than being derived from the
      input dimensionality.
    - The acquisition is the inherited analytic LogEI over the trust-region box,
      not Thompson sampling over a masked discrete candidate set.
    - Data not generated through the current embedding is least-squares
      projected into target space.

Reference:
    Papenmeier et al., "Increasing the Scope as You Learn: Adaptive Bayesian
    Optimization in Nested Subspaces", NeurIPS 2022.
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
    ValidationInfo,
    field_validator,
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
    """Scale lengthscales to mean 1, then unit geometric mean (volume-preserving).

    The geometric mean is taken as a product of per-element roots (the reference
    form). Forming ``weights.prod()`` first underflows to zero once target_dim
    reaches a few hundred, which would make every weight ``inf`` and silently
    widen the trust region to the whole domain.
    """
    weights = lengthscales / lengthscales.mean()
    return weights / weights.pow(1.0 / target_dim).prod()


class BAxUSTrustRegion(XoptBaseModel):
    """Serializable trust-region state, in normalized [0, 1] target-space units.

    ``update`` takes weighted objective values (MAXIMIZE -> +y, MINIMIZE -> -y),
    so higher is always better. ``best_value=None`` means no BO-phase
    evaluation has been observed yet.
    """

    target_dim: PositiveInt
    length: float = 0.8
    length_init: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    success_counter: int = 0
    success_tolerance: int = 3
    failure_tolerance: PositiveInt = Field(
        default=0,
        validate_default=True,
        description=(
            "Failures before shrinking; 0 derives ceil(target_dim / 2), re-derived "
            "on every BO-phase ingest (budget-aware when eval_budget is set)"
        ),
    )
    best_value: float | None = None
    restart_triggered: bool = False

    @field_validator("failure_tolerance", mode="before")
    @classmethod
    def _default_failure_tolerance(cls, value: Any, info: ValidationInfo) -> Any:
        """0 (the field default) means "derive from target_dim"."""
        if value:
            return value
        # absent only if target_dim itself failed validation; PositiveInt then rejects 0
        target_dim = info.data.get("target_dim")
        return math.ceil(target_dim / 2) if target_dim else value

    def update(self, y_weighted: torch.Tensor) -> None:
        """Adjust the trust region from a batch of weighted objective values."""
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
        elif self.failure_counter >= self.failure_tolerance:
            self.length /= 2.0
            self.failure_counter = 0

        self.best_value = y_max if best is None else max(best, y_max)

        if self.length < self.length_min:
            self.restart_triggered = True


class BAxUSEmbedding(XoptBaseModel):
    """Serializable sparse random embedding.

    ``matrix`` has shape (target_dim, input_dim) with exactly one signed unit
    entry per column. Both ``create`` and ``expand`` consume randomness; each
    takes an explicit seed so a run can be reproduced from a dump.
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

    def _as_array(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=np.float64)

    def lift(self, Z: np.ndarray) -> np.ndarray:
        """Lift from target subspace to full input space: ``X = Z @ S``."""
        return Z @ self._as_array()

    def project(self, X: np.ndarray) -> np.ndarray:
        """Project from full input space to target subspace via the pseudo-inverse."""
        return X @ np.linalg.pinv(self._as_array())

    def expand(
        self, new_bins_on_split: int, seed: int | None = None
    ) -> "BAxUSEmbedding":
        """Split each bin into up to ``new_bins_on_split + 1`` sub-bins, preserving signs.

        The dimensions contributing to a bin are randomly permuted before being
        split, matching the reference implementation - without it the split is a
        fixed function of column index, so dimensions that are adjacent in index
        and share a bin are never separated before full dimensionality. Pass a
        ``seed`` to keep an expansion reproducible across a dump -> construct
        round trip.
        """
        rng = np.random.default_rng(seed)
        S = self._as_array()
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
    """Bayesian Optimization with Adaptively Expanding Subspaces (BAxUS).

    Operates in a low-dimensional random embedding of the input space and
    gradually expands it when the trust region shrinks below a threshold.
    The acquisition (analytic LogEI) is inherited from
    ``ExpectedImprovementGenerator``; only the model/bounds seams are re-based
    into the embedded target space.

    This is a ``StateOwner``: the trust region and embedding are advanced once
    per ``generate`` call, so restoring a saved run reattaches data without
    replaying (and corrupting) the state that was just deserialized.
    """

    name = "baxus"
    supports_batch_generation: bool = False
    supports_single_objective: bool = True
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
        description="Initial target dimensionality of the embedding",
    )
    n_initial_sobol: int = Field(
        default=0,
        ge=0,
        description="Number of initial Sobol points (0 = auto-computed)",
    )
    length_init: float = Field(
        default=0.8,
        description="Initial trust-region side length",
    )
    new_bins_on_split: PositiveInt = Field(
        default=3,
        description="Number of new bins created per existing bin on expansion",
    )
    seed: int | None = Field(
        default=None,
        description=(
            "Random seed for the embedding and Sobol initialization. GP fitting and "
            "acquisition optimization use torch's global RNG and are not covered by it."
        ),
    )
    eval_budget: PositiveInt | None = Field(
        default=None,
        description=(
            "Total planned evaluations for the run, seed points included (the seed "
            "quota is subtracted internally). Enables the reference budget-aware "
            "failure tolerance; None keeps the ceil(target_dim / 2) heuristic."
        ),
    )
    embedding: BAxUSEmbedding = Field(
        description="Sparse random embedding (auto-created from vocs and seed when omitted)",
    )
    trust_region: BAxUSTrustRegion = Field(
        description="Trust-region state (auto-created when omitted)",
    )
    sobol_draws: int = Field(
        default=0,
        ge=0,
        description="Number of Sobol seed points drawn so far (used to resume the sequence)",
    )
    n_expansions: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of embedding expansions performed so far; combined with seed to "
            "keep each expansion's randomization reproducible across a round trip"
        ),
    )
    tr_observed_rows: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of finite-objective rows already folded into the trust region "
            "(prevents re-ingesting history on resume)"
        ),
    )
    # a plain default instance, matching BayesianGenerator: pydantic deep-copies
    # it per instance, and get_generator_defaults only understands `default`
    gp_constructor: SerializeAsAny[ModelConstructor] = Field(
        StandardModelConstructor(use_low_noise_prior=True),
        description=(
            "Constructor used to generate the target-space model. Defaults to a "
            "low-noise prior, matching the near-interpolating GP the reference "
            "trust-region logic assumes"
        ),
    )

    _sobol: SobolEngine | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _init_components(cls, values: Any) -> Any:
        """Fill in the vocs-derived defaults on fresh construction.

        Creates the embedding and trust region and sizes the Sobol seed quota
        from the embedding. A dump -> construct round trip supplies the
        components and skips their creation.
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
            length_init = float(values.get("length_init") or 0.8)
            values["trust_region"] = BAxUSTrustRegion(
                target_dim=emb_target_dim,
                length=length_init,
                length_init=length_init,
            )

        return values

    def model_post_init(self, context: Any, /) -> None:
        """Warn once that the reference expansion schedule is disabled.

        This hook runs exactly once per construction. A model validator of
        either mode would instead re-fire on every field assignment under
        ``validate_assignment``, repeating the warning on each trust-region
        update and embedding expansion.
        """
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
        """Sobol seed points first, then LogEI within the trust region.

        The BO phase defers to ``BayesianGenerator.generate`` through the
        ``get_training_data`` / ``_process_candidates`` / ``_get_torch_bounds``
        seams, so only the seed branch and the trust-region advance are
        BAxUS-specific. Only the BO phase records ``computation_time`` rows;
        the Sobol seed phase trains no model.
        """
        # the guard is repeated from the base call because the seed branch
        # below returns without ever reaching it
        if n_candidates > 1 and not self.supports_batch_generation:
            raise NotImplementedError(
                "This Bayesian algorithm does not currently support parallel candidate generation"
            )

        if len(self._finite_data()) < self.n_initial_sobol:
            self.n_candidates = n_candidates
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

        Replaces the base implementation wholesale: its discrete snapping and
        fixed-feature handling are keyed to vocs-space columns, and BAxUS
        rejects both options at construction.
        """
        x_raw = self._lift_to_vocs(candidates.detach().cpu().numpy())
        return convert_numpy_to_inputs(self.vocs, x_raw, include_constants=False)

    def _get_torch_bounds(self) -> torch.Tensor:
        """The full embedded target space, ``[-1, 1]^target_dim``.

        This deliberately re-points a base-class seam from vocs space into
        target space. It is safe because BAxUS overrides or rejects every other
        consumer: ``_get_optimization_bounds`` below, ``max_travel_distances``,
        and ``visualize_model``.
        """
        ones = torch.ones(self.embedding.target_dim, dtype=torch.double)
        return torch.stack([-ones, ones])

    def get_optimum(self) -> pd.DataFrame:
        """Select the best point given by the model's posterior mean.

        Optimizes over the full embedded target space and lifts the result to
        vocs space. The base implementation's constraint, fixed-feature,
        contextual, and discrete handling is inert here because BAxUS rejects
        all of those at construction.
        """
        # an expansion clears the model, and a freshly restored run has none;
        # retrain so the model always matches the current embedding
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

        Pure ingestion - the trust region is advanced in ``generate`` instead, so
        the state machine follows optimization iterations rather than how results
        were chunked into ``add_data`` calls.

        Non-finite objective rows stay in ``data`` but are excluded from the GP
        training set and the trust region. A batch missing the objective column
        entirely (what an evaluator returns for a failed evaluation under
        ``strict=False``) is ingested the same way.
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

        ``Xopt`` calls this instead of ``add_data`` when loading a saved run
        (see ``StateOwner``), so the deserialized embedding and trust region
        survive a checkpoint round trip intact.
        """
        self.data = data

    def _expansion_seed(self) -> int | None:
        """Seed for the next expansion, derived so a round trip reproduces it."""
        if self.seed is None:
            return None
        return self.seed + 104729 * (self.n_expansions + 1)

    def _advance_trust_region(self) -> None:
        """Fold newly observed BO-phase results into the trust region.

        Called once per ``generate``, mirroring how ``BayesianGenerator`` drives
        ``TurboController.update_state``. Sobol seed points never move the trust
        region (matches the reference), and ``tr_observed_rows`` records how much
        history has already been applied so a resumed run does not re-ingest it.
        """
        data = self._finite_data()
        n_finite = len(data)
        start = max(self.tr_observed_rows, self.n_initial_sobol)
        if n_finite <= start:
            return

        y = data[self._objective_name].to_numpy(dtype=np.float64)[start:]
        self.tr_observed_rows = n_finite

        # refresh lazily so an eval_budget assigned after construction takes effect
        self._refresh_failure_tolerance()
        self.trust_region.update(
            torch.tensor(self._objective_weight() * y, dtype=torch.double)
        )

        if self.trust_region.restart_triggered:
            previous_dim = self.embedding.target_dim
            self.embedding = self.embedding.expand(
                self.new_bins_on_split, seed=self._expansion_seed()
            )
            logger.info(
                "BAxUS: trust region too small (%.4e), expanded embedding from %d to %d dims",
                self.trust_region.length,
                previous_dim,
                self.embedding.target_dim,
            )
            self.n_expansions += 1
            self.trust_region = BAxUSTrustRegion(
                target_dim=self.embedding.target_dim,
                length=self.length_init,
                length_init=self.length_init,
                best_value=self.trust_region.best_value,
            )
            self._refresh_failure_tolerance()
            # the fitted model lives in the old target space
            self.model = None

    def train_model(
        self, data: pd.DataFrame | None = None, update_internal: bool = True
    ) -> Module:
        """Fit the GP on target-space projections of the finite-objective data."""
        data = data if data is not None else self._finite_data()
        if data.empty:
            raise ValueError("no data available to build model")

        Z = self._target_space(data)
        input_names = [f"z{i}" for i in range(self.embedding.target_dim)]
        objective_name = self._objective_name
        frame = pd.DataFrame(Z, columns=input_names)
        frame[objective_name] = data[objective_name].to_numpy(dtype=np.float64)

        # xopt's constructor downgrades a failed fit to a warning and returns an
        # untrained model (expected occasionally right after an embedding
        # expansion); re-emit captured warnings and log the degradation
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = self.gp_constructor.build_model(
                input_names=input_names,
                outcome_names=[objective_name],
                data=frame,
                input_bounds={name: [-1.0, 1.0] for name in input_names},
                **self.tkwargs,
            )
        for caught_warning in caught:
            warnings.warn_explicit(
                caught_warning.message,
                caught_warning.category,
                caught_warning.filename,
                caught_warning.lineno,
            )
        if any("Model fitting failed" in str(w.message) for w in caught):
            logger.warning(
                "BAxUS: GP fit failed at target_dim=%d - proceeding with an untrained model for this iteration",
                self.embedding.target_dim,
            )
        if update_internal:
            self.model = model
        return model

    def _get_optimization_bounds(self) -> torch.Tensor:
        """Trust-region bounds in target space [-1, 1]^target_dim.

        Centered on the incumbent, per-dimension widths scaled by lengthscale
        weights. Computed in [0, 1] units (BoTorch reference semantics for the
        length fields), then mapped to [-1, 1].
        """
        data = self._finite_data()
        y = torch.tensor(
            data[self._objective_name].to_numpy(dtype=np.float64),
            dtype=torch.double,
        )
        best_idx = int(torch.argmax(self._objective_weight() * y))

        Z = self._target_space(data)
        center01 = (torch.tensor(Z[best_idx], dtype=torch.double) + 1.0) / 2.0

        weights = self._lengthscale_weights(self.model)
        half_length = self.trust_region.length / 2.0
        lb01 = torch.clamp(center01 - half_length * weights, 0.0, 1.0)
        ub01 = torch.clamp(center01 + half_length * weights, 0.0, 1.0)
        return torch.stack([2.0 * lb01 - 1.0, 2.0 * ub01 - 1.0]).to(**self.tkwargs)

    def _refresh_failure_tolerance(self) -> None:
        """Set the trust region's failure tolerance, budget-aware when possible.

        Implements the reference schedule (Papenmeier et al., Alg. 1), pacing
        embedding expansions so the final split lands as the evaluation budget
        runs out; falls back to the ``ceil(target_dim / 2)`` heuristic without
        ``eval_budget``. At full dimensionality the tolerance widens to
        ``target_dim``. The planned split count is generalized to
        ``ceil(log_{b+1}(input_dim / d_init))`` to honor a user-chosen
        ``target_dim_init``.
        """
        region = self.trust_region
        # the embedding is the single source of truth for dimensionality
        target_dim = self.embedding.target_dim
        input_dim = self.embedding.input_dim
        if target_dim >= input_dim:
            region.failure_tolerance = target_dim
            return
        if self.eval_budget is None:
            # no budget: leave the ceil(target_dim / 2) heuristic the trust region
            # already derived for this target_dim
            return
        d_init = min(self.target_dim_init, input_dim)
        b = self.new_bins_on_split
        # The epsilon keeps exact powers of b+1 from rounding up on float noise.
        n_splits = math.ceil(math.log(input_dim / d_init, b + 1) - 1e-9)
        # each split multiplies dimensionality by (b + 1), so the planned splits
        # span a total growth factor of (b + 1) ** (n_splits + 1)
        growth = (b + 1) ** (n_splits + 1)
        # evaluations left for the BO phase, then this target_dim's share of them
        bo_budget = max(1, self.eval_budget - self.n_initial_sobol)
        split_budget = round(b * bo_budget * target_dim / (d_init * (growth - 1)))
        # max(1, ...): length_init == length_min would otherwise divide by zero
        halvings = max(
            1, math.floor(math.log(region.length_min / region.length_init, 0.5))
        )
        # spend that share evenly over the halvings needed to reach length_min
        region.failure_tolerance = min(
            target_dim, max(1, math.floor(split_budget / halvings))
        )

    @property
    def _objective_name(self) -> str:
        """The single objective this generator models (multi-objective is rejected)."""
        return self.vocs.objective_names[0]

    def _vocs_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-variable lower and upper vocs bounds, ordered as ``variable_names``."""
        lb, ub = get_variable_bounds_array(self.vocs)
        return lb, ub

    def _lift_to_vocs(self, z: np.ndarray) -> np.ndarray:
        """Lift target-space rows to raw vocs-space values.

        Lifts through the embedding, clips to ``[-1, 1]``, then scales to the
        per-variable vocs bounds.
        """
        x_norm = np.clip(self.embedding.lift(z), -1.0, 1.0)
        lb, ub = self._vocs_bounds()
        return lb + (x_norm + 1.0) / 2.0 * (ub - lb)

    def _objective_weight(self) -> float:
        """xopt objective weight: +1 for MAXIMIZE, -1 for MINIMIZE."""
        weights = set_botorch_weights(self.vocs)
        return float(weights[self.vocs.output_names.index(self._objective_name)])

    def _finite_objective_mask(self, frame: pd.DataFrame) -> np.ndarray:
        """Rows of ``frame`` with a finite objective; all-False if the column is absent.

        Note that ``vocs.extract_data(return_valid=True)`` cannot stand in here:
        it filters on feasibility, which is all-True without constraints.
        """
        column = frame.get(self._objective_name)
        if column is None:
            # every evaluation in the frame failed, so the column was never created
            return np.zeros(len(frame), dtype=bool)
        y = pd.to_numeric(column, errors="coerce").to_numpy(dtype=np.float64)
        return np.isfinite(y)

    def _finite_data(self) -> pd.DataFrame:
        """Rows of ``data`` with a finite objective - the GP training set."""
        if self.data is None or self.data.empty:
            return pd.DataFrame(
                columns=[*self.vocs.variable_names, self._objective_name]
            )
        return self.data[self._finite_objective_mask(self.data)]

    def _normalized_inputs(self, data: pd.DataFrame) -> np.ndarray:
        """Variable columns scaled to [-1, 1] per vocs bounds.

        Deliberately [-1, 1] rather than ``vocs.normalize_inputs``' [0, 1]: these
        feed a matrix product with the embedding, which is sign-symmetric.
        """
        lb, ub = self._vocs_bounds()
        X = data[self.vocs.variable_names].to_numpy(dtype=np.float64)
        return 2.0 * (X - lb) / (ub - lb) - 1.0

    def _target_space(self, data: pd.DataFrame) -> np.ndarray:
        """Variable columns of ``data`` projected into the embedded target space."""
        return self.embedding.project(self._normalized_inputs(data))

    def _draw_sobol_point(self) -> np.ndarray:
        """Next quasi-random seed point in target space [-1, 1]^target_dim.

        After a dump -> construct round trip the engine is rebuilt and
        fast-forwarded by ``sobol_draws`` (exact continuation needs a fixed
        ``seed``).
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

        Tries a gpytorch kernel lengthscale, then a duck-typed
        ``get_lengthscale_weights`` hook, then falls back to uniform weights
        (isotropic region).
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

        hook = getattr(single, "get_lengthscale_weights", None)
        weights = hook(target_dim) if hook is not None else None
        if weights is not None:
            return weights
        return torch.ones(target_dim, dtype=torch.double)
