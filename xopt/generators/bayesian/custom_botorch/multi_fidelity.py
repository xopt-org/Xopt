from typing import Callable, List, Optional, Union

import torch
from botorch.acquisition import InverseCostWeightedUtility
from botorch.acquisition.multi_objective import (
    MCMultiOutputObjective,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.models import AffineFidelityCostModel, GenericDeterministicModel
from botorch.models.model import Model
from botorch.sampling import MCSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    is_fully_bayesian,
    match_batch_shape,
    t_batch_mode_transform,
)
from torch import Tensor


class NMOMF(qNoisyExpectedHypervolumeImprovement):
    def __init__(
        self,
        model: Model,
        ref_point: Union[List[float], Tensor],
        X_baseline: Tensor,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        X_pending: Optional[Tensor] = None,
        eta: Optional[Union[Tensor, float]] = 1e-3,
        prune_baseline: bool = False,
        alpha: float = 0.0,
        cache_pending: bool = True,
        max_iep: int = 0,
        incremental_nehvi: bool = True,
        cache_root: bool = True,
        cost_call: Callable[[Tensor], Tensor] = None,
    ):
        super().__init__(
            model=model,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            eta=eta,
            X_pending=X_pending,
            prune_baseline=prune_baseline,
            alpha=alpha,
            cache_pending=cache_pending,
            max_iep=max_iep,
            incremental_nehvi=incremental_nehvi,
            cache_root=cache_root,
        )

        if cost_call is None:
            cost_model = AffineFidelityCostModel(
                fidelity_weights={-1: 1.0}, fixed_cost=1.0
            )
        else:
            cost_model = GenericDeterministicModel(cost_call)
        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
        self.cost_aware_utility = cost_aware_utility

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        X_full = torch.cat([match_batch_shape(self.X_baseline, X), X], dim=-2)
        # Note: it is important to compute the full posterior over `(X_baseline, X)`
        # to ensure that we properly sample `f(X)` from the joint distribution `
        # `f(X_baseline, X) ~ P(f | D)` given that we can already fixed the sampled
        # function values for `f(X_baseline)`.
        # TODO: improve efficiency by not recomputing baseline-baseline
        # covariance matrix.
        posterior = self.model.posterior(X_full)
        # Account for possible one-to-many transform and the MCMC batch dimension in
        # `SaasFullyBayesianSingleTaskGP`
        event_shape_lag = 1 if is_fully_bayesian(self.model) else 2
        n_w = (
            posterior._extended_shape()[X_full.dim() - event_shape_lag]
            // X_full.shape[-2]
        )
        q_in = X.shape[-2] * n_w
        self._set_sampler(q_in=q_in, posterior=posterior)
        samples = self._get_f_X_samples(posterior=posterior, q_in=q_in)

        # Add previous nehvi from pending points.
        nhv_gain = self._compute_qehvi(samples=samples, X=X) + self._prev_nehvi
        cost_weighted_qnehvi = self.cost_aware_utility(X=X, deltas=nhv_gain)
        return cost_weighted_qnehvi
