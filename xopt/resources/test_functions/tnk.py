import time
from random import random
from typing import Dict

import numpy as np

from xopt.vocs import VOCS

tnk_vocs = VOCS(
    **{
        "variables": {"x1": [0, 3.14159], "x2": [0, 3.14159]},
        "objectives": {"y1": "MINIMIZE", "y2": "MINIMIZE"},
        "constraints": {"c1": ["GREATER_THAN", 0], "c2": ["LESS_THAN", 0.5]},
        "constants": {"a": "dummy_constant"},
        # 'linked_variables': {'x9': 'x1'}
    }
)


# Pure number version
def TNK(individual):
    x1 = individual[0]
    x2 = individual[1]
    objectives = (x1, x2)
    constraints = (
        x1 ** 2 + x2 ** 2 - 1.0 - 0.1 * np.cos(16 * np.arctan2(x1, x2)),
        (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2,
    )
    return objectives, constraints


# labeled version
def evaluate_TNK(
    inputs: Dict,
    sleep=0,
    random_sleep=0,
    raise_probability=0,
    **params
):
    ind = [inputs["x1"], inputs["x2"]]
    objectives, constraints = TNK(ind)

    r = random()
    # Sleep for a bit
    time.sleep(sleep)
    time.sleep(r * random_sleep * 2)  # Average should be random_sleep

    if r < raise_probability:
        raise ValueError("intentioal TNK crash")

    outputs = {
        "y1": objectives[0],
        "y2": objectives[1],
        "c1": constraints[0],
        "c2": constraints[1],
        "some_array": np.array([1, 2, 3]),
    }

    return outputs
