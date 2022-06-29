from copy import deepcopy

import numpy as np
import pandas as pd

from xopt.resources.testing import TEST_VOCS_BASE
from xopt.vocs import ObjectiveEnum, VOCS


class TestVOCS(object):
    def test_init(self):
        from xopt.vocs import VOCS

        # test various configurations
        vocs = VOCS(
            variables={"x": [0, 1]},
            objectives={"f": "MINIMIZE"},
        )
        assert vocs.n_inputs == 1
        assert vocs.n_outputs == 1
        assert vocs.n_constraints == 0

    def test_from_yaml(self):
        Y = """
        variables:
          a: [0, 1e3] # Note that 1e3 usually parses as a str with YAML.
          b: [-1, 1]
        objectives:
          c: maximize
          d: minimize
        constraints:
          e: ['Less_than', 2]
          f: ['greater_than', 0]
        constants:
          g: 1234

        """

        vocs = VOCS.from_yaml(Y)
        assert vocs.constraint_names == ["e", "f"]
        assert vocs.variables == {"a": [0, 1e3], "b": [-1, 1]}
        assert vocs.objectives == {"c": "MAXIMIZE", "d": "MINIMIZE"}
        assert vocs.constants == {"g": 1234}

        assert vocs.objectives["c"] == ObjectiveEnum.MAXIMIZE
        assert vocs.objectives["d"] == ObjectiveEnum.MINIMIZE

        assert vocs.constraints["e"] == ["LESS_THAN", 2]
        assert vocs.constraints["f"] == ["GREATER_THAN", 0]

        assert vocs.n_inputs == 3
        assert vocs.n_outputs == 4

    def test_random_inputs(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        n_samples = 10
        data = pd.DataFrame(vocs.random_inputs(n_samples))
        assert data.shape == (n_samples, vocs.n_inputs)

    def test_serialization(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.json()

        vocs.variables["a"] = np.array([1, 2])
        vocs.json()

    def test_properties(self):
        vocs = deepcopy(TEST_VOCS_BASE)

        assert vocs.n_variables == 2
        assert vocs.n_inputs == 3
        assert vocs.n_outputs == 2
        assert vocs.n_constraints == 1
        assert vocs.n_objectives == 1
        assert vocs.variable_names == ["x1", "x2"]
        assert vocs.objective_names == ["y1"]

        # modify the vocs and retest
        vocs.variables = {name: vocs.variables[name] for name in ["x1"]}
        assert vocs.n_variables == 1
        assert vocs.n_inputs == 2
        assert vocs.n_outputs == 2
        assert vocs.variable_names == ["x1"]
