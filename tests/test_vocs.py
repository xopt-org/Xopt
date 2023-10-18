from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA
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

    def test_empty_objectives(self):
        VOCS(
            variables={"x": [0, 1]},
        )

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

        TEST_VOCS_BASE.random_inputs(5, include_constants=False)

    def test_serialization(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.model_dump_json()

        vocs.variables["a"] = np.array([1, 2])
        vocs.model_dump_json()

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

    def test_custom_bounds(self):
        vocs = deepcopy(TEST_VOCS_BASE)

        custom_bounds = {"x1": [0.5, 0.75], "x2": [7.5, 15.0]}

        random_input_data = vocs.random_inputs(100, custom_bounds=custom_bounds)
        random_input_data = pd.DataFrame(random_input_data)
        assert all(random_input_data["x1"] < 0.75)
        assert all(random_input_data["x1"] > 0.5)
        assert all(random_input_data["x2"] > 7.5)
        assert all(random_input_data["x2"] < 10.0)

    def test_duplicate_outputs(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        assert vocs.output_names == ["y1", "c1"]

        vocs.objectives = {"y1": "MAXIMIZE", "d1": "MINIMIZE"}
        vocs.observables = ["y1", "c1"]

        assert vocs.output_names == ["d1", "y1", "c1"]
        assert vocs.n_outputs == 3

    def test_convert_dataframe_to_inputs(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        test_data = TEST_VOCS_DATA

        with pytest.raises(ValueError):
            vocs.convert_dataframe_to_inputs(test_data)

        res = vocs.convert_dataframe_to_inputs(test_data[vocs.variable_names])
        assert "cnt1" in res

    def test_validate_input_data(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)

        # test good data
        test_vocs.validate_input_data(pd.DataFrame({"x1": 0.5, "x2": 1.0}, index=[0]))

        # test bad data
        with pytest.raises(ValueError):
            test_vocs.validate_input_data(
                pd.DataFrame({"x1": 0.5, "x2": 11.0}, index=[0])
            )

        with pytest.raises(ValueError):
            test_vocs.validate_input_data(
                pd.DataFrame({"x1": [-0.5, 2.5], "x2": [1.0, 11.0]})
            )

    def test_select_best(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        test_data = pd.DataFrame(
            {
                "x1": [0.1, 0.1, 0.1, 0.1],
                "x2": [0.1, 0.1, 0.1, 0.1],
                "c1": [1.0, 0.0, 1.0, 0.0],
                "y1": [0.5, 0.1, 1.0, 1.5],
            }
        )

        # test maximization
        vocs.objectives[vocs.objective_names[0]] = "MAXIMIZE"
        idx, val = vocs.select_best(test_data)
        assert idx == [2]
        assert val == [1.0]

        vocs.constraints = {}
        idx, val = vocs.select_best(test_data)
        assert idx == [3]
        assert val == [1.5]

        # test returning multiple best values -- sorted by best value
        idx, val = vocs.select_best(test_data, 2)
        assert np.allclose(idx, np.array([3, 2]))
        assert np.allclose(val, np.array([1.5, 1.0]))

        # test minimization
        vocs.objectives[vocs.objective_names[0]] = "MINIMIZE"
        vocs.constraints = {"c1": ["GREATER_THAN", 0.5]}
        idx, val = vocs.select_best(test_data)
        assert idx == [0]
        assert val == [0.5]

        vocs.constraints = {}
        idx, val = vocs.select_best(test_data)
        assert idx == 1
        assert val == 0.1
