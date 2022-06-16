from typing import Callable, List, Optional

import torch
from botorch.acquisition import (
    MCAcquisitionFunction,
)
from botorch.acquisition.objective import GenericMCObjective, PosteriorTransform
from botorch.models.model import Model
from botorch.utils import apply_constraints
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor


class FeasibilityObjective(GenericMCObjective):
    def __init__(
        self,
        constraints: List[Callable[[Tensor], Tensor]],
        infeasible_cost: float = 0.0,
        eta: float = 1e-3,
    ) -> None:
        def ones_callable(X):
            return torch.ones(X.shape[:-1])

        super().__init__(objective=ones_callable)
        self.constraints = constraints
        self.eta = eta
        self.register_buffer("infeasible_cost", torch.as_tensor(infeasible_cost))

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Evaluate the feasibility-weighted objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x m`-dim Tensors of
                samples from a model posterior.
            X: A `batch_shape x q x d`-dim tensor of inputs. Relevant only if
                the objective depends on the inputs explicitly.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of objective values
            weighted by feasibility (assuming maximization).
        """
        obj = super().forward(samples=samples)
        return apply_constraints(
            obj=obj,
            constraints=self.constraints,
            samples=samples,
            infeasible_cost=self.infeasible_cost,
            eta=self.eta,
        )


class ConstrainedMCAcquisitionFunction(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        base_acqusition: MCAcquisitionFunction,
        constraints: List[Callable],
        infeasible_cost=0.0,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        super().__init__(
            model=model,
            sampler=base_acqusition.sampler,
            objective=FeasibilityObjective(constraints),
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.infeasible_cost = infeasible_cost
        self.base_acqusition = base_acqusition

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.sampler(posterior)
        obj = self.objective(samples, X=X)

        # multiply the output of the base acquisition function by the feasibility
        return (self.base_acqusition(X) + self.infeasible_cost) * obj.max(dim=-1)[
            0
        ].mean(dim=0)
