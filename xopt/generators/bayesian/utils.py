import torch

from xopt.vocs import VOCS


def set_botorch_weights(weights, vocs: VOCS):
    """set weights to multiply xopt objectives for botorch objectives"""
    for idx, ele in enumerate(vocs.objective_names):
        if vocs.objectives[ele] == "MINIMIZE":
            weights[idx] = -1.0
        elif vocs.objectives[ele] == "MAXIMIZE":
            weights[idx] = 1.0

    return weights


def rectilinear_domain_union(A, B):
    """
    outputs a rectilinear domain that is the union of bounds A/B

    A.shape = (2,N)
    B.shape = (2,N)
    """
    assert A.shape[0] == 2
    assert A.shape == B.shape

    out_bounds = torch.clone(A)

    out_bounds[0, :] = torch.max(
        A[0, :],
        B[0, :],
    )
    out_bounds[1, :] = torch.min(
        A[1, :],
        B[1, :],
    )

    return out_bounds
