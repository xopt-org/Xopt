import math
from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from xopt import Evaluator, VOCS, Xopt
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
from xopt.generators.bayesian.bax.algorithms import GridOptimize
from xopt.generators.bayesian.bax_generator import BaxGenerator
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.turbo import (
    EntropyTurboController,
    OptimizeTurboController,
    SafetyTurboController,
)
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA


def sin_function(input_dict):
    x = input_dict["x"]
    return {"f": -10 * np.exp(-((x - np.pi) ** 2) / 0.01) + 0.5 * np.sin(5 * x)}


class TestTurbo(TestCase):
    def test_turbo_init(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.variables = {"x1": [0, 1]}

        state = OptimizeTurboController(test_vocs)
        assert state.dim == 1
        assert state.failure_tolerance == 2
        assert state.success_tolerance == 2
        assert state.minimize

        test_vocs.objectives[test_vocs.objective_names[0]] = "MAXIMIZE"
        state = OptimizeTurboController(test_vocs)
        assert state.dim == 1
        assert state.failure_tolerance == 2
        assert state.success_tolerance == 2
        assert not state.minimize

    @patch.multiple(BayesianGenerator, __abstractmethods__=set())
    def test_turbo_validation(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.variables = {"x1": [0, 1]}

        turbo_controller = OptimizeTurboController(test_vocs)
        BayesianGenerator(vocs=test_vocs, turbo_controller=turbo_controller)

        turbo_controller = {"name": "optimize", "length": 0.5}
        gen = BayesianGenerator(vocs=test_vocs, turbo_controller=turbo_controller)
        assert gen.turbo_controller.length == 0.5

        # turbo controller dict needs to have a name attribute
        with pytest.raises(ValueError):
            BayesianGenerator(
                vocs=test_vocs, turbo_controller={"bad_keyword": "result"}
            )

        # test specifying controller via string
        BayesianGenerator(vocs=test_vocs, turbo_controller="optimize")

        with pytest.raises(ValueError):
            BayesianGenerator(vocs=test_vocs, turbo_controller="bad_controller")

        # test not allowed generator type
        with pytest.raises(ValueError):
            BayesianGenerator(
                vocs=test_vocs, turbo_controller=EntropyTurboController(test_vocs)
            )

    @patch.multiple(BayesianGenerator, __abstractmethods__=set())
    def test_get_trust_region(self):
        # test in 1D
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.variables = {"x1": [0, 1]}

        gen = BayesianGenerator(vocs=test_vocs)
        gen.add_data(TEST_VOCS_DATA)
        gen.train_model()

        turbo_state = OptimizeTurboController(gen.vocs)
        turbo_state.update_state(gen)
        tr = turbo_state.get_trust_region(gen)
        assert tr[0].numpy() >= test_vocs.bounds[0]
        assert tr[1].numpy() <= test_vocs.bounds[1]

        # test in 2D
        test_vocs = deepcopy(TEST_VOCS_BASE)
        gen = BayesianGenerator(vocs=test_vocs)
        gen.add_data(TEST_VOCS_DATA)
        gen.train_model()

        turbo_state = OptimizeTurboController(gen.vocs)
        turbo_state.update_state(gen)
        tr = turbo_state.get_trust_region(gen)

        assert np.all(tr[0].numpy() >= test_vocs.bounds[0])
        assert np.all(tr[1].numpy() <= test_vocs.bounds[1])

    @patch.multiple(BayesianGenerator, __abstractmethods__=set())
    def test_restrict_data(self):
        # test in 1D
        test_vocs = deepcopy(TEST_VOCS_BASE)

        gen = BayesianGenerator(
            vocs=test_vocs, turbo_controller=OptimizeTurboController(test_vocs)
        )
        gen.add_data(TEST_VOCS_DATA)
        gen.train_model()
        gen.turbo_controller.update_state(gen)

        restricted_data = gen.get_training_data(gen.data)
        assert np.allclose(
            restricted_data["x1"].to_numpy(), np.array([0.45, 0.56, 0.67])
        )

    @patch.multiple(BayesianGenerator, __abstractmethods__=set())
    def test_with_constraints(self):
        # test in 1D
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.variables = {"x1": [0, 1]}
        test_vocs.constraints = {"c1": ["LESS_THAN", 0.0]}

        # test with valid data
        data = deepcopy(TEST_VOCS_DATA)
        data["c1"] = -10.0
        y_data = np.ones(10)
        y_data[5] = -1
        data["y1"] = y_data
        best_x = data["x1"].iloc[5]

        gen = BayesianGenerator(vocs=test_vocs)
        gen.add_data(data)
        gen.train_model()

        turbo_state = OptimizeTurboController(gen.vocs, failure_tolerance=5)
        turbo_state.update_state(gen)
        assert turbo_state.center_x == {"x1": best_x}
        assert turbo_state.success_counter == 0
        assert turbo_state.failure_counter == 1

        tr = turbo_state.get_trust_region(gen)
        assert tr[0].numpy() >= test_vocs.bounds[0]
        assert tr[1].numpy() <= test_vocs.bounds[1]

        # test a case where the last point is invalid
        new_data = deepcopy(gen.data)
        n_c = new_data["c1"].to_numpy()
        n_c[-1] = 1.0
        new_data["c1"] = n_c
        gen.add_data(new_data)
        turbo_state.update_state(gen)
        assert turbo_state.success_counter == 0
        assert turbo_state.failure_counter == 2

        # test will all invalid data
        data = deepcopy(TEST_VOCS_DATA)
        data["c1"] = 10.0
        y_data = np.ones(10)
        y_data[5] = -1
        data["y1"] = y_data

        gen = BayesianGenerator(vocs=test_vocs)
        gen.add_data(data)

        turbo_state = OptimizeTurboController(gen.vocs)
        with pytest.raises(RuntimeError):
            turbo_state.update_state(gen)

        # test best y value violates the constraint
        data = deepcopy(TEST_VOCS_DATA)
        c_data = -10.0 * np.ones(10)
        c_data[5] = 5.0
        data["c1"] = c_data
        y_data = np.ones(10)
        y_data[5] = -1
        y_data[6] = -0.8
        data["y1"] = y_data
        best_x = data["x1"].iloc[6]

        gen = BayesianGenerator(vocs=test_vocs)
        gen.add_data(data)

        turbo_state = OptimizeTurboController(gen.vocs, failure_tolerance=5)
        turbo_state.update_state(gen)
        assert turbo_state.center_x == {"x1": best_x}

        # test case where constraint violations give nan values for y
        data = deepcopy(TEST_VOCS_DATA)
        c_data = -10.0 * np.ones(10)
        c_data[5] = 5.0  # this point violates the constraint
        data["c1"] = c_data
        y_data = np.ones(10)
        y_data[5] = np.nan
        y_data[6] = -0.8
        data["y1"] = y_data
        best_x = data["x1"].iloc[6]

        gen = BayesianGenerator(vocs=test_vocs)
        gen.add_data(data)

        turbo_state = OptimizeTurboController(gen.vocs, failure_tolerance=5)
        turbo_state.update_state(gen)
        assert turbo_state.center_x == {"x1": best_x}

    @patch.multiple(BayesianGenerator, __abstractmethods__=set())
    def test_set_best_point(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)

        turbo_state = OptimizeTurboController(test_vocs)
        gen = BayesianGenerator(vocs=test_vocs)
        gen.add_data(TEST_VOCS_DATA)

        turbo_state.update_state(gen)
        best_value = TEST_VOCS_DATA[
            test_vocs.feasibility_data(TEST_VOCS_DATA)["feasible"]
        ].min()[test_vocs.objective_names[0]]
        best_point = TEST_VOCS_DATA.iloc[
            TEST_VOCS_DATA[test_vocs.feasibility_data(TEST_VOCS_DATA)["feasible"]][
                test_vocs.objective_names[0]
            ].idxmin()
        ][test_vocs.variable_names].to_dict()
        assert turbo_state.best_value == best_value
        assert turbo_state.center_x == best_point

        # test with maximization
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives[test_vocs.objective_names[0]] = "MAXIMIZE"

        turbo_state = OptimizeTurboController(test_vocs)
        assert not turbo_state.minimize
        gen = BayesianGenerator(vocs=test_vocs)
        gen.add_data(TEST_VOCS_DATA)

        turbo_state.update_state(gen)
        best_value = TEST_VOCS_DATA[
            test_vocs.feasibility_data(TEST_VOCS_DATA)["feasible"]
        ].max()[test_vocs.objective_names[0]]
        assert turbo_state.best_value == best_value

    @patch.multiple(BayesianGenerator, __abstractmethods__=set())
    def test_batch_turbo(self):
        # test in 1D
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.variables = {"x1": [0, 1]}
        test_vocs.constraints = {"c1": ["LESS_THAN", 0.0]}

        # test case where previous points were good
        data = deepcopy(TEST_VOCS_DATA)
        c_data = -10.0 * np.ones(10)
        data["c1"] = c_data
        y_data = np.ones(10)
        data["y1"] = y_data
        gen = BayesianGenerator(vocs=test_vocs)
        gen.add_data(data)

        turbo_state = OptimizeTurboController(test_vocs, failure_tolerance=5)
        turbo_state.update_state(gen, 2)
        assert turbo_state.success_counter == 1

    def test_in_generator(self):
        vocs = VOCS(
            variables={"x": [0, 2 * math.pi]},
            objectives={"f": "MINIMIZE"},
        )

        def sin_function(input_dict):
            x = input_dict["x"]
            return {"f": -10 * np.exp(-((x - np.pi) ** 2) / 0.01) + 0.5 * np.sin(5 * x)}

        evaluator = Evaluator(function=sin_function)
        generator = UpperConfidenceBoundGenerator(
            vocs=vocs, turbo_controller="optimize"
        )
        X = Xopt(evaluator=evaluator, generator=generator, vocs=vocs)

        X.evaluate_data(pd.DataFrame({"x": [3.0, 1.75, 2.0]}))

        # determine trust region from gathered data
        X.generator.train_model()
        X.generator.turbo_controller.update_state(X.generator)
        X.generator.turbo_controller.get_trust_region(X.generator)

        for i in range(2):
            X.step()

    @patch.multiple(BayesianGenerator, __abstractmethods__=set())
    def test_safety(self):
        test_vocs = VOCS(
            variables={"x": [0, 2 * math.pi]},
            objectives={"f": "MINIMIZE"},
            constraints={"c": ["LESS_THAN", 0]},
        )

        test_data = pd.DataFrame(
            {"x": [0.5, 1.0, 1.5], "f": [1.0, 1.0, 1.0], "c": [-1.0, -1.0, 1.0]}
        )
        sturbo = SafetyTurboController(vocs=test_vocs)
        gen = BayesianGenerator(vocs=test_vocs)
        gen.add_data(test_data)

        sturbo.update_state(gen)

        assert sturbo.center_x == {"x": 0.75}
        assert sturbo.failure_counter == 1

        # test batch case where all data is good
        test_data2 = pd.DataFrame(
            {"x": [0.5, 1.0, 1.5], "f": [1.0, 1.0, 1.0], "c": [-1.0, -1.0, -1.0]}
        )

        gen = BayesianGenerator(vocs=test_vocs)
        gen.add_data(test_data2)

        sturbo.update_state(gen, previous_batch_size=3)
        assert sturbo.success_counter == 1
        assert sturbo.failure_counter == 0

        # test batch case where only some of the data is good
        # note default `min_feasible_fraction` is 0.75
        test_data3 = pd.DataFrame(
            {"x": [0.5, 1.0, 1.5], "f": [1.0, 1.0, 1.0], "c": [-1.0, 1.0, -1.0]}
        )
        gen = BayesianGenerator(vocs=test_vocs)
        gen.add_data(test_data3)

        sturbo.update_state(gen, previous_batch_size=3)
        assert sturbo.success_counter == 0
        assert sturbo.failure_counter == 1

    def test_serialization(self):
        vocs = VOCS(
            variables={"x": [0, 2 * math.pi]},
            objectives={"f": "MINIMIZE"},
        )

        evaluator = Evaluator(function=sin_function)
        for name in ["optimize", "safety"]:
            generator = UpperConfidenceBoundGenerator(vocs=vocs, turbo_controller=name)
            X = Xopt(evaluator=evaluator, generator=generator, vocs=vocs)

            yaml_str = X.yaml()
            X2 = Xopt.from_yaml(yaml_str)
            assert X2.generator.turbo_controller.name == name

    def test_entropy_turbo(self):
        # define variables and function objectives
        vocs = VOCS(
            variables={"x": [0, 2 * math.pi]},
            observables=["y1"],
        )

        def sin_function(input_dict):
            return {"y1": np.sin(input_dict["x"])}

        # Prepare BAX algorithm and generator options
        algorithm = GridOptimize(n_mesh_points=10)  # NOTE: default is to minimize

        # construct BAX generator
        generator = BaxGenerator(
            vocs=vocs,
            algorithm=algorithm,
            turbo_controller=EntropyTurboController(
                vocs, success_tolerance=2, failure_tolerance=2
            ),
        )

        # construct evaluator
        evaluator = Evaluator(function=sin_function)

        # construct Xopt optimizer
        X = Xopt(evaluator=evaluator, generator=generator, vocs=vocs)

        X.random_evaluate(3)

        for i in range(2):
            X.step()

        # test not allowed generator type
        with pytest.raises(ValueError):
            BaxGenerator(
                vocs=vocs,
                algorithm=algorithm,
                turbo_controller=OptimizeTurboController(vocs),
            )
