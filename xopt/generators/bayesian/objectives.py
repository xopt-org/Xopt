from abc import ABC, abstractmethod
from functools import partial
from typing import Optional, Callable, List

import torch
from botorch.acquisition import (
    LinearMCObjective,
    MCAcquisitionObjective,
)
from botorch.acquisition.multi_objective import WeightedMCMultiOutputObjective
from botorch.sampling import get_sampler
from torch import Tensor

from xopt import VOCS

from xopt.generators.bayesian.custom_botorch.constrained_acquisition import (
    FeasibilityObjective,
)
from xopt.generators.bayesian.utils import set_botorch_weights


class CustomXoptObjective(MCAcquisitionObjective, ABC):
    """
    Custom objective function wrapper for use in Bayesian generators.

    Attributes:
    -----------
    vocs : VOCS
        The VOCS (Variables, Objectives, Constraints, Statics) object.

    Methods:
    --------
    forward(samples: Tensor, X: Optional[Tensor] = None) -> Tensor
        Evaluate the objective on the samples.
    """

    def __init__(self, vocs: VOCS, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocs = vocs

    @abstractmethod
    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Evaluate the objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x m`-dim Tensors of
                samples from a model posterior.
            X: A `batch_shape x q x d`-dim tensor of inputs. Relevant only if
                the objective depends on the inputs explicitly.

        Returns:
            Tensor: A `sample_shape x batch_shape x q`-dim Tensor of objective
            values (assuming maximization).

        This method is usually not called directly, but via the objectives.

        Example:
            >>> # `__call__` method:
            >>> samples = sampler(posterior)
            >>> outcome = mc_obj(samples)
        """
        pass


def feasibility(
    X: Tensor,
    model: torch.nn.Module,
    vocs: VOCS,
    posterior_transform: Optional[Callable] = None,
) -> Tensor:
    """
    Calculate the feasibility of the given points.

    Parameters:
    -----------
    X : Tensor
        The input tensor.
    model : torch.nn.Module
        The model used for Bayesian Optimization.
    vocs : VOCS
        The VOCS (Variables, Objectives, Constraints, Statics) object.
    posterior_transform : Optional[Callable], optional
        The posterior transform, by default None.

    Returns:
    --------
    Tensor
        The feasibility values.
    """
    constraints = create_constraint_callables(vocs)
    posterior = model.posterior(X=X, posterior_transform=posterior_transform)

    sampler = get_sampler(
        model.posterior(X),
        sample_shape=torch.Size([512]),
    )
    samples = sampler(posterior)
    objective = FeasibilityObjective(constraints)
    return torch.mean(objective(samples, X), dim=0)


def constraint_function(Z: Tensor, vocs: VOCS, name: str) -> Tensor:
    """
    Create constraint function.

    Constraint functions should return negative values for feasible values and
    positive values for infeasible values.

    Parameters:
    -----------
    Z : Tensor
        The input tensor.
    vocs : VOCS
        The VOCS (Variables, Objectives, Constraints, Statics) object.
    name : str
        The name of the constraint.

    Returns:
    --------
    Tensor
        The constraint values.
    """
    output_names = vocs.output_names
    constraint = vocs.constraints[name]

    if constraint[0] == "LESS_THAN":
        return Z[..., output_names.index(name)] - constraint[1]
    elif constraint[0] == "GREATER_THAN":
        return -(Z[..., output_names.index(name)] - constraint[1])


def create_constraint_callables(vocs: VOCS) -> Optional[List[Callable]]:
    """
    Create a list of constraint callables.

    Parameters:
    -----------
    vocs : VOCS
        The VOCS (Variables, Objectives, Constraints, Statics) object.

    Returns:
    --------
    Optional[List[Callable]]
        A list of constraint callables, or None if there are no constraints.
    """
    if vocs.constraints is not None:
        constraint_names = vocs.constraint_names
        constraint_callables = []
        for name in constraint_names:
            constraint_callables += [
                partial(
                    constraint_function,
                    vocs=vocs,
                    name=name,
                )
            ]
        return constraint_callables

    else:
        return None


def create_mc_objective(vocs: VOCS) -> LinearMCObjective:
    """
    Create a monte carlo objective object.

    Parameters:
    -----------
    vocs : VOCS
        The VOCS (Variables, Objectives, Constraints, Statics) object.

    Returns:
    --------
    LinearMCObjective
        The objective object.
    """
    weights = set_botorch_weights(vocs)
    objective = LinearMCObjective(weights=weights)
    return objective


def create_mobo_objective(vocs: VOCS) -> WeightedMCMultiOutputObjective:
    """
    Create the multi-objective Bayesian optimization objective.

    BoTorch assumes maximization, so we need to negate any objectives that have
    the minimize keyword and zero out anything that is a constraint.

    Parameters:
    -----------
    vocs : VOCS
        The VOCS (Variables, Objectives, Constraints, Statics) object.

    Returns:
    --------
    WeightedMCMultiOutputObjective
        The multi-objective Bayesian optimization objective.
    """
    output_names = vocs.output_names
    if vocs.n_outputs == vocs.n_objectives:
        objective_indices = None
        weights = set_botorch_weights(vocs)
    else:
        objective_indices = [output_names.index(name) for name in vocs.objectives]
        weights = set_botorch_weights(vocs)[objective_indices]

    # Note that objective_indices gets registered as buffer with no device specified by botorch
    # If it is None, a lot of checks are skipped so we do this for performance
    objective = WeightedMCMultiOutputObjective(
        weights=weights,
        outcomes=objective_indices,
        num_outcomes=vocs.n_objectives,
    )
    return objective
