from functools import partial
from typing import Callable, List, Optional

import torch
from botorch.acquisition import ConstrainedMCObjective, GenericMCObjective
from botorch.acquisition.multi_objective import WeightedMCMultiOutputObjective
from botorch.utils import apply_constraints
from torch import Tensor


def constraint_function(Z, vocs, index, quantile_cutoff=0.0):
    """
    create constraint function
    - if a distribution of samples has a quantile level, given by `quantile_cutoff`,
    that is infeasiable penalize the entire set of samples to make all infeasible
    """
    n_objectives = len(vocs.objectives)

    # quantile test
    output = Z[..., n_objectives + index] + 5.0 * (
        torch.quantile(
            Z[..., n_objectives + index], quantile_cutoff, dim=0, keepdim=True
        )
        > 0
    )
    return output


def create_constraint_callables(vocs, quantile_cutoff=0.5):
    if vocs.constraints is not None:
        constraint_names = list(vocs.constraints.keys())
        n_constraints = len(constraint_names)
        constraint_callables = []
        for i in range(n_constraints):
            constraint_callables += [
                partial(
                    constraint_function,
                    vocs=vocs,
                    index=i,
                    quantile_cutoff=quantile_cutoff,
                )
            ]
        return constraint_callables

    else:
        return None


def create_mc_objective(vocs):
    """
    create the objective object

    """
    weights = torch.zeros(vocs.n_outputs).double()
    for idx, ele in enumerate(vocs.objective_names):
        if vocs.objectives[ele] == "MINIMIZE":
            weights[idx] = -1.0
        else:
            weights[idx] = 1.0

    def obj_callable(Z):
        return torch.matmul(Z, weights.reshape(-1, 1)).squeeze(-1)

    return GenericMCObjective(obj_callable)


def create_constrained_mc_objective(vocs, quantile_cutoff=0.0):
    """
    create the objective object

    """
    weights = torch.zeros(vocs.n_outputs).double()
    for idx, ele in enumerate(vocs.objective_names):
        if vocs.objectives[ele] == "MINIMIZE":
            weights[idx] = -1.0
        else:
            weights[idx] = 1.0

    def obj_callable(Z):
        return torch.matmul(Z, weights.reshape(-1, 1)).squeeze(-1)

    constraint_callables = create_constraint_callables(vocs, quantile_cutoff)

    constrained_obj = ConstrainedMCObjective(
        objective=obj_callable, constraints=constraint_callables, infeasible_cost=100.0
    )

    return constrained_obj


def create_mobo_objective(vocs):
    """
    botorch assumes maximization so we need to negate any objectives that have
    minimize keyword and zero out anything that is a constraint
    """
    n_objectives = len(vocs.objectives)
    weights = torch.zeros(n_objectives).double()

    for idx, ele in enumerate(vocs.objectives):
        if vocs.objectives[ele] == "MINIMIZE":
            weights[idx] = -1.0
        else:
            weights[idx] = 1.0

    return WeightedMCMultiOutputObjective(
        weights, outcomes=list(range(n_objectives)), num_outcomes=n_objectives
    )


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
            infeasible_cost=0.0,
            eta=self.eta,
        )
