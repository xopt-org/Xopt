from typing import Dict

import numpy as np

from xopt.vocs import VOCS

modified_tnk_vocs = VOCS(
    **{
        "variables": {"x1": [0, 3.14159], "x2": [0, 3.14159]},
        "objectives": {"y1": "MINIMIZE"},
        "constraints": {"c1": ["GREATER_THAN", 0], "c2": ["LESS_THAN", 0.5]},
        "constants": {"a": "dummy_constant"},
        # 'linked_variables': {'x9': 'x1'}
    }
)


# Pure number version
def modified_TNK(individual):
    x1 = individual[0]
    x2 = individual[1]
    objectives = (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2
    constraints = (
        x1**2 + x2**2 - 1.0 - 0.1 * np.cos(16 * np.arctan2(x1, x2)),
        (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2,
    )
    return objectives, constraints


# labeled version
def evaluate_modified_TNK(inputs: Dict, extra_option="abc", **params):
    ind = [inputs["x1"], inputs["x2"]]
    objectives, constraints = modified_TNK(ind)
    outputs = {
        "y1": objectives,
        "c1": constraints[0],
        "c2": constraints[1],
        "some_array": np.array([1, 2, 3]),
    }

    return outputs
