from typing import Dict

import numpy as np

from xopt.vocs import VOCS

dtlz2_vocs = VOCS(
    **{
        "variables": {"x1": [0, 1], "x2": [0, 1], "x3": [0, 1]},
        "objectives": {"y1": "MINIMIZE", "y2": "MINIMIZE"},
        "constraints": {},
        "constants": {"a": "dummy_constant"},
    }
)
dtlz2_reference_point = {"y1": 1.1, "y2": 1.1}


# From BoTorch
def DTLZ2(X: np.ndarray) -> np.ndarray:
    assert X.shape[1] == len(dtlz2_vocs.variables)
    k = X.shape[1] - 2 + 1
    X_m = X[..., -k:]
    g_X = ((X_m - 0.5) ** 2).sum(axis=-1)
    g_X_plus1 = 1 + g_X
    fs = []
    pi_over_2 = np.pi / 2
    for i in range(2):
        idx = 2 - 1 - i
        f_i = g_X_plus1.copy()
        f_i *= np.cos(X[..., :idx] * pi_over_2).prod(axis=-1)
        if i > 0:
            f_i *= np.sin(X[..., idx] * pi_over_2)
        fs.append(f_i)
    return np.stack(fs, axis=-1)


def evaluate_DTLZ2(inputs: Dict, **params):
    ind = np.array([inputs["x1"], inputs["x2"], inputs["x3"]])
    # TODO: make function input consistent from all generators
    if ind.ndim == 1:
        # MOBO yields floats
        ind = ind[np.newaxis, :]
    else:
        # Random generator yields length-1 numpy arrays
        ind = ind.T
    objectives = DTLZ2(ind)

    outputs = {
        "y1": objectives[0, 0],
        "y2": objectives[0, 1],
    }

    return outputs
