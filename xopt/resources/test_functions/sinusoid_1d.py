import numpy as np

from xopt.vocs import VOCS

sinusoid_vocs = VOCS(
    **{
        "variables": {"x1": [0, 1.75 * 3.14159]},
        "objectives": {"y1": "MINIMIZE"},
        "constraints": {"c1": ["GREATER_THAN", 0]},
    }
)


# labeled version
def evaluate_sinusoid(inputs: dict):
    outputs = {
        "y1": np.sin(inputs["x1"]),
        "c1": 10 * np.sin(inputs["x1"]) - 9.5 + np.sin(7.0 * inputs["x1"]),
    }
    return outputs
