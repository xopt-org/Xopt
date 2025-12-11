"""
eval_fun.py - Define the evaluation function

This file will be imported by xopt and supply the optimizer with the evaluation function to call
"""

import numpy as np


def eval_fun(in_dict: dict, n: int = 30) -> dict:
    """
    The function is ZDT3 from [1]. It is implemented using numpy vectorized operations to allow its use in
    quick example problems.

    [1] Zitzler, E., Deb, K., & Thiele, L. (2000). Comparison of Multiobjective Evolutionary Algorithms: Empirical Results.
        Evolutionary Computation, 8(2), 173–195.

    Parameters
    ----------
    in_dict : dict
        The dictionary of decision variables x1, x2, ..., x30
    n : int
        Number of decision variables, by default 30

    Returns
    -------
    dict
        The dictionary of objectives (f1, f2)
    """
    # Unpack the decision var dict
    x = np.array([in_dict[f"x{idx}"] for idx in range(1, n + 1)])

    # Calculate objectives
    g = 1 + 9 * np.sum(x[1:], axis=0) / (n - 1)
    ret = {
        "f1": x[0].tolist(),
        "f2": (
            g * (1 - np.sqrt(x[0] / g) - x[0] / g * np.sin(10 * np.pi * x[0]))
        ).tolist(),
    }
    return ret
