import torch
from botorch.acquisition import ConstrainedMCObjective


def create_constrained_mc_objective(vocs):
    """
    create the objective object
    """
    n_objectives = len(vocs.objectives)
    n_constraints = len(vocs.constraints)
    weights = torch.zeros(n_objectives + n_constraints).double()
    for idx, ele in enumerate(vocs.objectives):
        if vocs.objectives[ele] == 'MINIMIZE':
            weights[idx] = -1.0
        else:
            weights[idx] = 1.0

    def obj_callable(Z):
        return torch.matmul(Z, weights.reshape(-1, 1)).squeeze(-1)

    def constraint_callable(Z):
        out = torch.ones(*Z.shape[:-1]).to(Z)
        for j, cele in enumerate(vocs.constraints):
            if vocs.constraints[cele][0] == 'GREATER_THAN':
                out = out * (
                        Z[..., n_objectives + j] - vocs.constraints[cele][1]
                ).to(out)
            elif vocs.constraints[cele][0] == 'LESS_THAN':
                out = out * -(
                        Z[..., n_objectives + j] - vocs.constraints[cele][1]
                ).to(out)

        return out

    constrained_obj = ConstrainedMCObjective(
        objective=obj_callable,
        constraints=[constraint_callable],
    )

    return constrained_obj


