from functools import partial

import torch
from botorch.acquisition import GenericMCObjective
from botorch.acquisition.multi_objective import WeightedMCMultiOutputObjective


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


def create_mc_objective(vocs, tkwargs):
    """
    create the objective object

    """
    weights = torch.zeros(vocs.n_outputs, **tkwargs)
    for idx, ele in enumerate(vocs.objective_names):
        if vocs.objectives[ele] == "MINIMIZE":
            weights[idx] = -1.0
        elif vocs.objectives[ele] == "MAXIMIZE":
            weights[idx] = 1.0

    def obj_callable(Z):
        return torch.matmul(Z, weights.reshape(-1, 1)).squeeze(-1)

    return GenericMCObjective(obj_callable)


def create_mobo_objective(vocs, tkwargs):
    """
    botorch assumes maximization so we need to negate any objectives that have
    minimize keyword and zero out anything that is a constraint
    """
    n_objectives = len(vocs.objectives)
    weights = torch.zeros(n_objectives).to(**tkwargs)

    for idx, ele in enumerate(vocs.objectives):
        if vocs.objectives[ele] == "MINIMIZE":
            weights[idx] = -1.0
        elif vocs.objectives[ele] == "MAXIMIZE":
            weights[idx] = 1.0

    return WeightedMCMultiOutputObjective(
        weights, outcomes=list(range(n_objectives)), num_outcomes=n_objectives
    )


def create_momf_objective(vocs, tkwargs):
    """
    create multi-objective multi-fidelity objective assuming that the last axis is
    the fidelity parameter
    botorch assumes maximization so we need to negate any objectives that have
    minimize keyword and zero out anything that is a constraint
    """
    n_objectives = len(vocs.objectives) + 1
    weights = torch.zeros(n_objectives).to(**tkwargs)

    for idx, ele in enumerate(vocs.objectives):
        if vocs.objectives[ele] == "MINIMIZE":
            weights[idx] = -1.0
        elif vocs.objectives[ele] == "MAXIMIZE":
            weights[idx] = 1.0

    # set fidelity objective (which is always maximize)
    weights[-1] = 1.0
    return WeightedMCMultiOutputObjective(
        weights, outcomes=list(range(n_objectives)), num_outcomes=n_objectives
    )
