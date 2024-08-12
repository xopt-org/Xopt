import warnings
from typing import Callable, List, Optional

import torch
from botorch.acquisition import MCAcquisitionFunction
from botorch.acquisition.objective import GenericMCObjective, PosteriorTransform
from botorch.models.model import Model
from botorch.sampling import MCSampler
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
        def ones_callable(Z, X=None):
            return torch.ones(Z.shape[:-1])

        super().__init__(objective=ones_callable)
        self.constraints = constraints
        self.eta = eta
        self.register_buffer("infeasible_cost", torch.as_tensor(infeasible_cost))

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Evaluate the feasibility-weighted objective on the samples.

        Parameters
        ----------
            samples: A `sample_shape x batch_shape x q x m`-dim Tensors of
                samples from a model posterior.
            X: A `batch_shape x q x d`-dim tensor of inputs. Relevant only if
                the objective depends on the inputs explicitly.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of objective values
            weighted by feasibility (assuming maximization).
        """
        obj = super().forward(samples=samples).to(samples)
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
        base_acquisition: MCAcquisitionFunction,
        constraints: List[Callable],
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        # make it consistent with botorch constrained EHVI
        if constraints is None:
            constraints = []

        super().__init__(
            model=model,
            sampler=sampler,
            objective=FeasibilityObjective(constraints),
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.base_acquisition = base_acquisition

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        if self.objective.constraints:
            posterior = self.model.posterior(
                X=X, posterior_transform=self.posterior_transform
            )
            samples = self.get_posterior_samples(posterior)
            obj = self.objective(samples, X=X)

            # check base acquisition function for negative values
            base_acq_val = self.base_acquisition(X)
            min_value = torch.min(base_acq_val)
            if min_value < 0.0:
                warnings.warn(
                    "The base acquisition function has negative values and a softplus transformation will be "
                    "applied. This may cause numerical issues for large negative values."
                )
                base_val = torch.nn.functional.softplus(base_acq_val, beta=1)
            else:
                base_val = base_acq_val

            # multiply the output of the base acquisition function by the feasibility
            return base_val * obj.max(dim=-1)[0].mean(dim=0)
        else:
            return self.base_acquisition(X)
