from functools import partial

import torch
from botorch.acquisition import ConstrainedMCObjective
from botorch.acquisition.multi_objective import WeightedMCMultiOutputObjective


def constraint_function(Z, vocs=None, index=-1):
    n_objectives = len(vocs.objectives)
    constraint_names = list(vocs.constraints.keys())

    name = constraint_names[index]
    if vocs.constraints[name][0] == "GREATER_THAN":
        return vocs.constraints[name][1] - Z[..., n_objectives + index]
    elif vocs.constraints[name][0] == "LESS_THAN":
        return Z[..., n_objectives + index] - vocs.constraints[name][1]
    else:
        raise RuntimeError(
            f"constraint type {vocs.constraints[name][0]} not "
            "implemented for constrained bayes opt"
        )


def create_constraint_callables(vocs):
    if vocs.constraints is not None:
        constraint_names = list(vocs.constraints.keys())
        n_constraints = len(constraint_names)
        constraint_callables = []
        for i in range(n_constraints):
            constraint_callables += [partial(constraint_function, vocs=vocs, index=i)]
        return constraint_callables

    else:
        return None


def create_constrained_mc_objective(vocs):
    """
    create the objective object

    NOTE: we assume that corresponding model outputs are ordered according to vocs
    ordering
    """
    n_objectives = len(vocs.objectives)
    if vocs.constraints is not None:
        constraint_names = list(vocs.constraints.keys())
    else:
        constraint_names = []
    n_constraints = len(constraint_names)
    weights = torch.zeros(n_objectives + n_constraints).double()
    for idx, ele in enumerate(vocs.objectives):
        if vocs.objectives[ele] == "MINIMIZE":
            weights[idx] = -1.0
        else:
            weights[idx] = 1.0

    def obj_callable(Z):
        return torch.matmul(Z, weights.reshape(-1, 1)).squeeze(-1)

    constraint_callables = create_constraint_callables(vocs)

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
