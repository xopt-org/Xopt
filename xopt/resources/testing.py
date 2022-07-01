import numpy as np
import pandas as pd

from xopt.vocs import VOCS


def xtest_callable(input_dict: dict, a=0) -> dict:
    assert isinstance(input_dict, dict)
    x1 = input_dict["x1"]
    x2 = input_dict["x2"]

    y1 = x2
    c1 = x1
    return {"y1": y1, "c1": c1}


TEST_VOCS_BASE = VOCS(
    **{
        "variables": {"x1": [0, 1.0], "x2": [0, 10.0]},
        "objectives": {"y1": "MINIMIZE"},
        "constraints": {"c1": ["GREATER_THAN", 0.5]},
        "constants": {"cnt1": 1.0},
    }
)

cnames = (
    list(TEST_VOCS_BASE.variables.keys())
    + list(TEST_VOCS_BASE.objectives.keys())
    + list(TEST_VOCS_BASE.constraints.keys())
)
test_init_data = {
    "x1": np.random.rand(10),
    "x2": np.random.rand(10) * 10.0,
    "cnt1": 1.0,
}
test_init_data.update(xtest_callable(test_init_data))

TEST_VOCS_DATA = pd.DataFrame(test_init_data)

TEST_YAML = """
xopt:
    asynch: True
generator:
    name: random

evaluator:
    function: xopt.resources.testing.xtest_callable
    function_kwargs:
        a: 5

vocs:
    variables:
        x1: [0, 3.14159]
        x2: [0, 3.14159]
    objectives: {y1: MINIMIZE, y2: MINIMIZE}
    constraints:
        c1: [GREATER_THAN, 0]
        c2: ['LESS_THAN', 0.5]
"""
