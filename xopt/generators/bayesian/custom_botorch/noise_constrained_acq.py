from typing import Callable

import torch
from botorch.acquisition import (
    SampleReducingMCAcquisitionFunction,
)
from botorch.acquisition.monte_carlo import SampleReductionProtocol
from botorch.acquisition.objective import (
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.acquisition.utils import (
    repeat_to_match_aug_dim,
)
from botorch.exceptions.warnings import legacy_ei_numerics_warning
from botorch.models.model import Model
from botorch.models.transforms import Standardize
from botorch.sampling.base import MCSampler
from botorch.utils import t_batch_mode_transform
from botorch.utils.objective import compute_smoothed_feasibility_indicator
from botorch.utils.transforms import concatenate_pending_points
from torch import Tensor


class UncertaintyConstrainedAcquisitionFunction(SampleReducingMCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        sampler: MCSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
        sample_reduction: SampleReductionProtocol = torch.mean,
        q_reduction: SampleReductionProtocol = torch.amax,
        constraints: list[Callable[[Tensor], Tensor]] | None = None,
        eta: Tensor | float = 1e-3,
        fat: list[bool | None] | bool = False,
        variance_limit: dict[int, float] = None,
        variance_eta: float | dict[int, float] = None,
        variance_min_mult: float | dict[int, float] = 1e-3,
        variance_normalize: bool | dict[int, bool] = False,
    ) -> None:
        super().__init__(
            model,
            sampler,
            objective,
            posterior_transform,
            X_pending,
            sample_reduction,
            q_reduction,
            constraints,
            eta,
            fat,
        )
        self.variance_limit = variance_limit
        if variance_limit is None:
            assert variance_eta is None, (
                "If no variance limit is provided, variance_eta must also be None."
            )

        if isinstance(variance_eta, float):
            variance_eta = {k: variance_eta for k in variance_limit.keys()}
        else:
            assert isinstance(variance_eta, dict)
            assert set(variance_eta.keys()) == set(variance_limit.keys())
        self._variance_eta = variance_eta

        if isinstance(variance_min_mult, float):
            variance_min_mult = {k: variance_min_mult for k in variance_limit.keys()}
        else:
            assert isinstance(variance_min_mult, dict)
            assert set(variance_min_mult.keys()) == set(variance_limit.keys())
        self._variance_min_mult = variance_min_mult

        if isinstance(variance_normalize, bool):
            variance_normalize = {k: variance_normalize for k in variance_limit.keys()}
        else:
            assert isinstance(variance_normalize, dict)
            assert set(variance_normalize.keys()) == set(variance_limit.keys())
        self.variance_normalize = variance_normalize

    def _non_reduced_forward(self, X: Tensor) -> Tensor:
        """Compute the constrained acquisition values at the MC-sample, q level.

        Args:
            X: A `batch_shape x q x d` Tensor of t-batches with `q` `d`-dim
                design points each.

        Returns:
            A Tensor with shape `sample_sample x batch_shape x q`.
        """
        samples, obj = self._get_samples_and_objectives(X)
        samples = repeat_to_match_aug_dim(target_tensor=samples, reference_tensor=obj)
        acqval = self._sample_forward(obj)
        acqval = self._apply_constraints(acqval=acqval, samples=samples)
        acqval = self._apply_constraints_vl(acqval=acqval, samples=samples, obj=obj)
        return acqval

    def _apply_constraints_vl(
        self, acqval: Tensor, samples: Tensor, obj: Tensor, stdev: Tensor = None
    ) -> Tensor:
        """Multiplies the acquisition utility by noise constraint indicators.

        Args:
            acqval: `sample_shape x batch_shape x q`-dim acquisition utility values.
            samples: `sample_shape x batch_shape x q x m`-dim posterior samples.
            obj: `sample_shape x batch_shape x q`-dim objective values (index 0).

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of acquisition utility values
                multiplied by a smoothed constraint indicator per sample.


        Note that reductions happen after acqval samples are computed and penalized by constraints:
        self._sample_reduction = partial(sample_reduction, dim=sample_dim) (typically dim 0)
        self._q_reduction = partial(q_reduction, dim=-1)
        self._sample_reduction(self._q_reduction(non_reduced_acqval))

        We apply noise constraint on the samples, not on the variance.
        TODO: check if applying after reduction is more efficient.
        """
        if self.variance_limit is not None:
            vl_callables = []
            vl_eta = []
            for idx, threshold in self.variance_limit.items():
                # Apply variance thresholding - note that mean reduction will apply later
                def variance_callable(X: Tensor) -> Tensor:
                    if idx == 0:
                        if stdev is not None:
                            var = stdev
                        else:
                            mean = obj.mean(dim=0)
                            var = (obj - mean).abs()
                    else:
                        # var = X[..., idx].var(dim=0)
                        mean = X[..., idx].mean(dim=0)
                        var = (X[..., idx] - mean).abs()

                    if self.variance_normalize[idx]:
                        # # samples_standardized, _ = otransform(X[..., idx:idx+1])
                        # normvar = var / (otransform.stdvs * otransform.stdvs)
                        # # var = samples_standardized.var(dim=0)
                        idx_model = self.model.models[idx]
                        otransform = idx_model.outcome_transform
                        if not isinstance(otransform, Standardize):
                            raise ValueError(
                                f"Expected Standardize transform, got {type(otransform)}"
                            )
                        assert not otransform.training
                        normvar = var * otransform.stdvs
                        result = normvar - threshold
                    else:
                        result = var - threshold
                    # callable takes 1 x q x m tensor of reduced sample variance and returns 1 x q tensor of weights
                    return result

                vl_callables.append(variance_callable)
                vl_eta.append(self._variance_eta[idx])

            ind = compute_smoothed_feasibility_indicator(
                constraints=vl_callables,
                samples=samples,
                eta=torch.tensor(vl_eta),
                log=self._log,
                fat=False,
            )
            ind = torch.clamp(ind, self._variance_min_mult[idx], None)
            if self._log:
                acqval = acqval.add(ind)
            else:
                acqval = acqval.mul(ind)
        return acqval

    @concatenate_pending_points
    @t_batch_mode_transform()
    def _forward_constraint_c(self, X):
        """Return only the penalty from applying regular constraints"""
        samples, obj = self._get_samples_and_objectives(X)
        samples = repeat_to_match_aug_dim(target_tensor=samples, reference_tensor=obj)
        acqval = torch.ones_like(obj)
        acqval = self._apply_constraints(acqval=acqval, samples=samples)
        acqval = self._sample_reduction(self._q_reduction(acqval))
        return acqval

    @concatenate_pending_points
    @t_batch_mode_transform()
    def _forward_constraint_nc(self, X):
        """Return only the penalty from applying noise constraints"""
        samples, obj = self._get_samples_and_objectives(X)
        samples = repeat_to_match_aug_dim(target_tensor=samples, reference_tensor=obj)
        acqval = torch.ones_like(obj)
        acqval = self._apply_constraints_vl(acqval=acqval, samples=samples, obj=obj)
        acqval = self._sample_reduction(self._q_reduction(acqval))
        return acqval

    @concatenate_pending_points
    @t_batch_mode_transform()
    def _forward_pure(self, X):
        """Return only the acquisition utility without constraints"""
        samples, obj = self._get_samples_and_objectives(X)
        acqval = self._sample_forward(obj)
        acqval = self._sample_reduction(self._q_reduction(acqval))
        return acqval


class UCqExpectedImprovement(UncertaintyConstrainedAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        sampler: MCSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
        constraints: list[Callable[[Tensor], Tensor]] | None = None,
        eta: Tensor | float = 1e-3,
        variance_limit: dict[int, float] = None,
        variance_eta: float | dict[int, float] = None,
        variance_min_mult: float | dict[int, float] = 1e-3,
        variance_normalize: bool | dict[int, bool] = False,
    ) -> None:
        legacy_ei_numerics_warning(legacy_name=type(self).__name__)
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            constraints=constraints,
            eta=eta,
            variance_limit=variance_limit,
            variance_eta=variance_eta,
            variance_min_mult=variance_min_mult,
            variance_normalize=variance_normalize,
        )
        self.register_buffer("best_f", torch.as_tensor(best_f))

    def _sample_forward(self, obj: Tensor) -> Tensor:
        return (obj - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)


class UCqPosteriorVariance(UncertaintyConstrainedAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        sampler: MCSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
        constraints: list[Callable[[Tensor], Tensor]] | None = None,
        eta: Tensor | float = 1e-3,
        variance_limit: dict[int, float] = None,
        variance_eta: float | dict[int, float] = None,
        variance_min_mult: float | dict[int, float] = 1e-3,
        variance_normalize: bool | dict[int, bool] = False,
    ) -> None:
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            constraints=constraints,
            eta=eta,
            variance_limit=variance_limit,
            variance_eta=variance_eta,
            variance_min_mult=variance_min_mult,
            variance_normalize=variance_normalize,
        )

    def _non_reduced_forward(self, X: Tensor) -> Tensor:
        """Override standard method to optimize variance computation"""
        samples, obj = self._get_samples_and_objectives(X)
        # obj: A `sample_shape x batch_shape x q`-dim Tensor of MC objective values.
        samples = repeat_to_match_aug_dim(target_tensor=samples, reference_tensor=obj)

        mean = obj.mean(dim=0)
        stdev = acqval = (obj - mean).abs()
        # acqval: `sample_shape x batch_shape x q`-dim Tensor of acquisition utility values.

        acqval = self._apply_constraints(acqval=acqval, samples=samples)
        acqval = self._apply_constraints_vl(acqval=acqval, samples=samples, stdev=stdev)
        return acqval

    def _sample_forward(self, obj: Tensor) -> Tensor:
        raise Exception("Do not call directly, use _non_reduced_forward instead.")
