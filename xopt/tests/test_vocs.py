import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
import yaml

from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA
from xopt.vocs import (
    convert_dataframe_to_inputs,
    cumulative_optimum,
    denormalize_inputs,
    get_variable_data,
    normalize_inputs,
    random_inputs,
    select_best,
    grid_inputs,
    get_objective_data,
    validate_input_data,
    extract_data,
)
from gest_api.vocs import (
    VOCS,
    ContinuousVariable,
    DiscreteVariable,
    MinimizeObjective,
    ExploreObjective,
    LessThanConstraint,
)


class TestVOCS(object):
    def test_init(self):
        # test various configurations
        vocs = VOCS(
            variables={
                "x": [0, 1],
                "y": {0, 1, 2},
            },
            objectives={"f": "MINIMIZE"},
        )
        assert vocs.n_inputs == 2
        assert vocs.n_outputs == 1
        assert vocs.n_constraints == 0

    def test_variable_validation(self):
        with pytest.raises(ValidationError):
            VOCS(
                variables={"x": [1, 0]},
            )

    def test_empty_objectives(self):
        VOCS(
            variables={"x": [0, 1]},
        )

    def test_output_names(self):
        test_vocs = VOCS(
            variables={"x": [0, 1]},
            objectives={"y1": "MINIMIZE"},
            constraints={"c1": ["GREATER_THAN", 0], "c2": ["LESS_THAN", 0]},
        )
        assert test_vocs.output_names == ["y1", "c1", "c2"]

    def test_constraint_specification(self):
        good_constraint_list = [
            ["LESS_THAN", 0],
            ["GREATER_THAN", 0],
            ["less_than", 0],
            ["greater_than", 0],
        ]

        for ele in good_constraint_list:
            VOCS(variables={"x": [0, 1]}, constraints={"c": ele})

        bad_constraint_list = [
            ["LESS_THAN"],
            ["GREATER_TAN", 0],
            [0, "less_than"],
            ["greater_than", "ahc"],
        ]

        for ele in bad_constraint_list:
            with pytest.raises(ValidationError):
                VOCS(variables={"x": [0, 1]}, constraints={"c": ele})

    def test_from_yaml(self):
        Y = """
        constants: {}
        constraints:
            c:
              type: LessThanConstraint
              value: 0.0
        objectives:
            z: MINIMIZE
            zzz: EXPLORE
        variables:
            x:
              type: ContinuousVariable
              domain:
              - 0.0
              - 6.283185307179586
            y:
              type: DiscreteVariable
              values:
                - 0.0
                - 1.0
                - 2.0

        """
        vocs = VOCS(**yaml.safe_load(Y))
        assert vocs.constraint_names == ["c"]
        assert isinstance(vocs.variables["x"], ContinuousVariable)
        assert isinstance(vocs.variables["y"], DiscreteVariable)
        assert vocs.objectives["z"] == MinimizeObjective
        assert vocs.objectives["zzz"] == ExploreObjective
        assert isinstance(vocs.constraints["c"], LessThanConstraint)

        assert vocs.n_inputs == 2

    def test_random_inputs(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        n_samples = 10
        data = pd.DataFrame(random_inputs(vocs, n_samples))
        assert data.shape == (n_samples, vocs.n_inputs)

        test_inputs = random_inputs(vocs, 5, include_constants=False)
        assert len(test_inputs) == 5

        test_inputs = random_inputs(TEST_VOCS_BASE)
        assert isinstance(test_inputs[0]["x1"], float)

    def test_serialization(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.model_dump_json()

        vocs.variables["a"] = ContinuousVariable(domain=[0.0, 1.0])
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

    def test_grid_inputs(self):
        # Define a sample VOCS object
        vocs = VOCS(
            variables={"x1": [0.0, 1.0], "x2": [0.0, 1.0]},
            constraints={},
            objectives={},
            constants={"c1": 5.0},
            observables=[],
        )

        # Test with default parameters
        n = 5
        grid = grid_inputs(vocs, n=n)
        assert isinstance(grid, pd.DataFrame)
        assert grid.shape == (n**2, 3)  # 2 variables + 1 constant
        assert "x1" in grid.columns
        assert "x2" in grid.columns
        assert "c1" in grid.columns
        assert np.all(grid["c1"] == 5.0)

        # Test with custom bounds
        custom_bounds = {"x1": [0.2, 0.8], "x2": [0.1, 0.9]}
        grid = grid_inputs(vocs, n=n, custom_bounds=custom_bounds)
        assert isinstance(grid, pd.DataFrame)
        assert grid.shape == (n**2, 3)  # 2 variables + 1 constant
        assert "x1" in grid.columns
        assert "x2" in grid.columns
        assert "c1" in grid.columns
        assert np.all(grid["c1"] == 5.0)
        assert np.all(grid["x1"] >= 0.2) and np.all(grid["x1"] <= 0.8)
        assert np.all(grid["x2"] >= 0.1) and np.all(grid["x2"] <= 0.9)

        # Test with invalid custom bounds
        invalid_custom_bounds = {
            "x1": [1.2, 0.8],  # Invalid bounds
            "x2": [0.1, 0.9],
        }
        with pytest.raises(ValueError):
            grid_inputs(vocs, n=n, custom_bounds=invalid_custom_bounds)

        # Test with include_constants=False
        grid = grid_inputs(vocs, n=n, include_constants=False)
        assert isinstance(grid, pd.DataFrame)
        assert grid.shape == (n**2, 2)  # 2 variables
        assert "x1" in grid.columns
        assert "x2" in grid.columns
        assert "c1" not in grid.columns

        # Test with different number of points for each variable
        n_dict = {"x1": 3, "x2": 4}
        grid = grid_inputs(vocs, n=n_dict)
        assert isinstance(grid, pd.DataFrame)
        assert grid.shape == (3 * 4, 3)  # 2 variables + 1 constant
        assert "x1" in grid.columns
        assert "x2" in grid.columns
        assert "c1" in grid.columns
        assert np.all(grid["c1"] == 5.0)

    def test_random_sampling_custom_bounds(self):
        vocs = deepcopy(TEST_VOCS_BASE)

        custom_bounds = {"x1": [0.5, 0.75], "x2": [7.5, 15.0]}

        with pytest.warns(RuntimeWarning):
            random_input_data = random_inputs(vocs, 100, custom_bounds=custom_bounds)

        random_input_data = pd.DataFrame(random_input_data)
        assert all(random_input_data["x1"] < 0.75)
        assert all(random_input_data["x1"] > 0.5)
        assert all(random_input_data["x2"] > 7.5)
        assert all(random_input_data["x2"] < 10.0)

        # test custom bounds within the vocs domain -- no warnings should be raised
        in_domain_custom_bounds = {"x1": [0.5, 0.75], "x2": [0.5, 0.75]}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            random_inputs(vocs, 100, custom_bounds=in_domain_custom_bounds)

        # test wrong type
        with pytest.raises(TypeError):
            random_inputs(vocs, 100, custom_bounds=1)

        # test custom bounds entirely outside the vocs domain or specified incorrectly
        bad_custom_bounds = [
            {"x1": [10.0, 10.75], "x2": [7.5, 15.0]},
            {"x1": [0.75, 0.5], "x2": [7.5, 15.0]},
        ]
        for ele in bad_custom_bounds:
            with pytest.raises(ValueError):
                random_inputs(vocs, 100, custom_bounds=ele)

        custom_bounds = {
            k: [v.domain[0] + 0.01, v.domain[1] - 0.01]
            for k, v in TEST_VOCS_BASE.variables.items()
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            random_inputs(vocs, 3, custom_bounds=custom_bounds)

        custom_bounds = {
            k: [v.domain[0] - 0.01, v.domain[1] - 0.01]
            for k, v in TEST_VOCS_BASE.variables.items()
        }
        with pytest.warns(RuntimeWarning):
            random_inputs(vocs, 3, custom_bounds=custom_bounds)

    def test_duplicate_outputs(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        assert vocs.output_names == ["y1", "c1"]

        vocs.objectives = {"y1": "MAXIMIZE", "d1": "MINIMIZE"}
        vocs.observables = ["y1", "c1"]

        assert vocs.output_names == ["y1", "d1", "c1"]
        assert vocs.n_outputs == 3

    def test_variable_data(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        test_data = TEST_VOCS_DATA

        res = get_variable_data(vocs, test_data)
        assert np.array_equal(res.to_numpy(), test_data.loc[:, ["x1", "x2"]].to_numpy())

    def test_objective_data(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        test_data = TEST_VOCS_DATA
        test_data["y2"] = 1.0

        res = get_objective_data(vocs, test_data)
        assert np.array_equal(res.to_numpy(), test_data.loc[:, ["y1"]].to_numpy())

        vocs.objectives.update({"y2": "MAXIMIZE"})
        res = get_objective_data(vocs, test_data)
        assert np.array_equal(
            res.to_numpy(),
            test_data.loc[:, ["y1", "y2"]].to_numpy() * np.array([1, -1]),
        )

        test_data2 = test_data.drop(columns=["y1"])
        res = get_objective_data(vocs, test_data2)
        assert np.array_equal(
            res.to_numpy(),
            test_data.loc[:, ["y1", "y2"]].to_numpy() * np.array([np.inf, -1]),
        )

        # test using object type inside test data
        test_data["y2"] = test_data["y2"].astype(np.dtype(object))
        get_objective_data(vocs, test_data)

    def test_convert_dataframe_to_inputs(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        test_data = TEST_VOCS_DATA

        with pytest.raises(ValueError):
            convert_dataframe_to_inputs(vocs, test_data)

        res = convert_dataframe_to_inputs(vocs, test_data[vocs.variable_names])
        assert "constant1" in res

    def test_validate_input_data(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)

        # test good data
        validate_input_data(test_vocs, pd.DataFrame({"x1": 0.5, "x2": 1.0}, index=[0]))

        # test bad data
        with pytest.raises(ValueError):
            validate_input_data(
                test_vocs, pd.DataFrame({"x1": 0.5, "x2": 11.0}, index=[0])
            )

        with pytest.raises(ValueError):
            validate_input_data(
                test_vocs, pd.DataFrame({"x1": [-0.5, 2.5], "x2": [1.0, 11.0]})
            )

    def test_select_best(self):
        test_data = pd.DataFrame(
            {
                "x1": [0.1, 0.1, 0.1, 0.1],
                "x2": [0.1, 0.1, 0.1, 0.1],
                "c1": [1.0, 0.0, 1.0, 0.0],
                "y1": [0.5, 0.1, 1.0, 1.5],
            }
        )

        test_data_obj = deepcopy(test_data).astype(object)

        for ele in [test_data, test_data_obj]:
            # test maximization
            vocs = deepcopy(TEST_VOCS_BASE)

            vocs.objectives[vocs.objective_names[0]] = "MAXIMIZE"
            idx, val, _ = select_best(vocs, ele)
            assert idx == [2]
            assert val == [1.0]

            vocs.constraints = {}
            idx, val, _ = select_best(vocs, ele)
            assert idx == [3]
            assert val == [1.5]

            # test returning multiple best values -- sorted by best value
            idx, val, _ = select_best(vocs, ele, 2)
            assert np.allclose(idx, np.array([3, 2]))
            assert np.allclose(val, np.array([1.5, 1.0]))

            # test minimization
            vocs.objectives[vocs.objective_names[0]] = "MINIMIZE"
            vocs.constraints = {"c1": ["GREATER_THAN", 0.5]}
            idx, val, _ = select_best(vocs, ele)
            assert idx == [0]
            assert val == [0.5]

            vocs.constraints = {}
            idx, val, _ = select_best(vocs, ele)
            assert idx == 1
            assert val == 0.1

            # test error handling
            with pytest.raises(RuntimeError):
                select_best(vocs, pd.DataFrame())

            vocs.constraints = {"c1": ["GREATER_THAN", 10.5]}
            with pytest.raises(RuntimeError):
                select_best(vocs, pd.DataFrame())

    @pytest.mark.filterwarnings("ignore: All-NaN axis encountered")
    def test_cumulative_optimum(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        obj_name = vocs.objective_names[0]
        test_data = pd.DataFrame(
            {
                obj_name: [np.nan, 0.0, -0.4, 0.6, np.nan, -0.7],
                vocs.constraint_names[0]: [1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            }
        )
        vocs.objectives = {}
        with pytest.raises(RuntimeError):
            _ = cumulative_optimum(vocs, test_data)

        vocs.objectives = {obj_name: "MINIMIZE"}
        assert cumulative_optimum(vocs, pd.DataFrame()).empty

        cumulative_minimum = cumulative_optimum(vocs, test_data)
        assert np.array_equal(
            cumulative_minimum[f"best_{obj_name}"].values,
            np.array([np.nan, np.nan, -0.4, -0.4, -0.4, -0.7]),
            equal_nan=True,
        )

        vocs.objectives[obj_name] = "MAXIMIZE"
        cumulative_maximum = cumulative_optimum(vocs, test_data)
        assert np.array_equal(
            cumulative_maximum[f"best_{obj_name}"].values,
            np.array([np.nan, np.nan, -0.4, 0.6, 0.6, 0.6]),
            equal_nan=True,
        )

    def test_normalize_inputs(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        test_data = pd.DataFrame({"x1": [0.5, 0.2], "x2": [0.5, 0.75]})
        normed_data = normalize_inputs(vocs, test_data)
        assert normed_data["x1"].to_list() == [0.5, 0.2]
        assert normed_data["x2"].to_list() == [0.05, 0.075]

        assert np.equal(
            denormalize_inputs(vocs, normed_data)[vocs.variable_names].to_numpy(),
            test_data[vocs.variable_names].to_numpy(),
        ).all()

        test_data.pop("x1")
        normed_data = normalize_inputs(vocs, test_data)
        assert normed_data["x2"].to_list() == [0.05, 0.075]

        assert np.equal(
            denormalize_inputs(vocs, normed_data)[test_data.columns].to_numpy(),
            test_data[test_data.columns].to_numpy(),
        ).all()

        test_data = pd.DataFrame({"x1": 0.5}, index=[0])
        normed_data = normalize_inputs(vocs, test_data)
        assert normed_data["x1"].to_list() == [0.5]

        assert np.equal(
            denormalize_inputs(vocs, normed_data)[test_data.columns].to_numpy(),
            test_data[test_data.columns].to_numpy(),
        ).all()

        # test with extra data
        test_data = pd.DataFrame({"x1": [0.5, 0.2], "x2": [0.5, 0.75], "a": [1, 1]})
        normed_data = normalize_inputs(vocs, test_data)
        assert {"x1", "x2"} == set(normed_data.columns)

        assert np.equal(
            denormalize_inputs(vocs, normed_data)[vocs.variable_names].to_numpy(),
            test_data[vocs.variable_names].to_numpy(),
        ).all()

    def test_extract_data(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        test_data = pd.DataFrame(
            {
                "x1": [0.1, 0.2, 0.4, 0.4],
                "x2": [0.1, 0.2, 0.3, 0.2],
                "c1": [1.0, 0.0, 1.0, 0.0],
                "y1": [0.5, 0.1, 1.0, 1.5],
            }
        )

        # test default functionality
        data = extract_data(vocs, test_data)
        assert set(data[0].keys()) == {"x1", "x2"}
        assert set(data[1].keys()) == {"y1"}
        assert set(data[2].keys()) == {"c1"}
        for ele in data[:-1]:  # ignore observable data
            assert len(ele) == 4

        # test return_valid functionality
        data = extract_data(vocs, test_data, return_valid=True)
        assert data[0]["x1"].to_list() == [0.1, 0.4]
        assert data[1]["y1"].to_list() == [0.5, 1.0]

        for ele in data[:-1]:  # ignore observable data
            assert len(ele) == 2

        # test w/o constraints
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.constraints = {}
        data = extract_data(vocs, test_data)
        assert len(data[0]) == 4
        assert data[2].empty
