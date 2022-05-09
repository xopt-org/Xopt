import numpy as np
import pandas as pd

from xopt.vocs import VOCS, ObjectiveEnum
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA


class TestVOCS(object):
    def test_init(self):
        from xopt.vocs import VOCS

        vocs = VOCS()

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

        assert vocs.objectives['c'] == ObjectiveEnum.MAXIMIZE
        assert vocs.objectives['d'] == ObjectiveEnum.MINIMIZE

        assert vocs.constraints['e'] == ['LESS_THAN', 2]
        assert vocs.constraints['f'] == ['GREATER_THAN', 0]

        assert vocs.n_inputs == 3
        assert vocs.n_outputs == 4

    def test_random_inputs(self):
        vocs = TEST_VOCS_BASE
        n_samples = 10
        data = pd.DataFrame(vocs.random_inputs(n_samples))
        assert data.shape == (n_samples, vocs.n_inputs)

    def test_numpy_inputs(self):
        vocs = TEST_VOCS_BASE
        vocs.variables['a'] = np.array([1, 2])

        vocs.json()

