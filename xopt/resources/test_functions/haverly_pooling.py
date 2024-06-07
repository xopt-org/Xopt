import numpy as np

from xopt import VOCS

"""
C.A. Floudas, P.M. Pardalos
A Collection of Test Problems for Constrained
Global Optimization Algorithms, vol. 455,
Springer Science & Business Media (1990)
"""

variables = {
    "x1": [0, 100],
    "x2": [0, 200],
    "x3": [0, 100],
    "x4": [0, 100],
    "x5": [0, 100],
    "x6": [0, 100],
    "x7": [0, 200],
    "x8": [0, 100],
    "x9": [0, 200],
}
objectives = {"f": "MAXIMIZE"}
tol = 0.5
constraints = {
    "h1": ["LESS_THAN", tol],
    "h2": ["LESS_THAN", tol],
    "h3": ["LESS_THAN", tol],
    "h4": ["LESS_THAN", tol],
    # "g1": ["LESS_THAN", 0.0],
    # "g2": ["LESS_THAN", 0.0]
}

vocs_haverly = VOCS(variables=variables, objectives=objectives, constraints=constraints)


def evaluate_haverly(input_dict):
    x1 = input_dict["x1"]
    x2 = input_dict["x2"]
    x3 = input_dict["x3"]
    x4 = input_dict["x4"]
    x5 = input_dict["x5"]
    x6 = input_dict["x6"]
    x7 = input_dict["x7"]
    x8 = input_dict["x8"]
    x9 = input_dict["x9"]

    result = {
        "f": 9 * x1 + 15 * x2 - 6 * x3 - 16 * x4 - 10 * (x5 + x6),
        "h1": np.abs(x7 + x8 - x4 - x3) / 100,
        "h2": np.abs(x1 - x5 - x7) / 100,
        "h3": np.abs(x2 - x6 - x8) / 100,
        "h4": np.abs(x9 * x7 + x9 * x8 - 3 * x3 - x4) / 10000,
        "g1": (x9 * x7 + 2 * x5 - 2.5 * x1) / 1000,
        "g2": (x9 * x8 + 2 * x6 - 1.5 * x2) / 1000,
    }
    result["f"] = result["f"] / 1000.0

    return result
