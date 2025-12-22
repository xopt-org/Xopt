from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from scipy.optimize import minimize

from xopt import VOCS, Xopt
from xopt.vocs import get_variable_data, select_best
from xopt.errors import SeqGeneratorError
from xopt.generators.sequential.neldermead import NelderMeadGenerator
from xopt.resources.test_functions.ackley_20 import ackley, vocs as ackleyvocs
from xopt.resources.test_functions.rosenbrock import (
    rosenbrock,
    rosenbrock2_vocs as rbvocs,
)
from xopt.resources.testing import TEST_VOCS_BASE


def compare(X, X2):
    """Compare two Xopt objects"""
    y = json.loads(X.json())
    y2 = json.loads(X2.json())
    y.pop("data")
    y2.pop("data")
    assert y == y2
    # For unclear reasons, column order changes on reload....
    data = X.data.drop(["xopt_runtime", "xopt_error"], axis=1)
    data2 = X2.data.drop(["xopt_runtime", "xopt_error"], axis=1)
    # On reload, index is not a range index anymore!
    pd.testing.assert_frame_equal(data, data2, check_index_type=False)


def eval_f_linear_pos(x):
    return {"y1": np.sum([x**2 for x in x.values()])}


def eval_f_linear_neg(x):
    return {"y1": -np.sum([x**2 for x in x.values()])}


class TestNelderMeadGenerator:
    def test_simplex_generate_multiple_points(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.constraints = {}
        gen = NelderMeadGenerator(vocs=test_vocs)

        # Try to generate multiple samples
        with pytest.raises(SeqGeneratorError):
            gen.generate(2)

    def test_simplex_generate(self):
        """test simplex without providing an initial point -- started from point in data"""
        YAML = """
        generator:
            name: neldermead
            adaptive: true
        evaluator:
            function: xopt.resources.test_functions.rosenbrock.evaluate_rosenbrock
        vocs:
            variables:
                x0: [-5, 5]
                x1: [-5, 5]
            objectives: {y: MINIMIZE}
        """
        X = Xopt.from_yaml(YAML)
        X.random_evaluate(1)

        for _ in range(2):
            X.step()

        # test reloading
        state = X.json()
        X2 = Xopt.model_validate(json.loads(state))
        X2.step()

    def test_simplex_forced_init(self):
        """test to make sure that a re-loaded simplex generator works the same as the normal one at each step"""

        YAML = """
        generator:
            name: neldermead
            initial_point: {x0: -1, x1: -1}
            adaptive: true
        evaluator:
            function: xopt.resources.test_functions.rosenbrock.evaluate_rosenbrock
        vocs:
            variables:
                x0: [-5, 5]
                x1: [-5, 5]
            objectives: {y: MINIMIZE}
        """

        # test where we first random evaluate a point before starting simplex -- simplex will still start with the initial point
        X = Xopt.from_yaml(YAML)
        X.random_evaluate(1)
        assert not X.generator.is_active
        assert X.generator._last_candidate is None
        X.step()
        assert X.generator.is_active
        assert X.generator._last_candidate is not None
        assert X.generator._initial_simplex is None
        X.step()
        state = X.json()
        X2 = Xopt.model_validate(json.loads(state))
        X2.step()

        # test where we first random evaluate multiple points before starting simplex
        X = Xopt.from_yaml(YAML)
        X.random_evaluate(3)
        assert X.generator._initial_simplex is not None
        X.step()
        assert X.generator._initial_simplex is not None
        X.step()
        state = X.json()
        X2 = Xopt.model_validate(json.loads(state))
        X2.step()

        # test where we start simplex immediately but then try to add a random evaluation in the middle
        X = Xopt.from_yaml(YAML)
        X.step()
        # print(X.generator.data, X.generator.current_state.ngen)
        assert X.generator._initial_simplex is None
        assert X.generator.current_state.astg == 0
        with pytest.raises(SeqGeneratorError):
            X.random_evaluate(1)
        X.step()
        X.step()
        assert X.generator._initial_simplex is None
        assert X.generator.current_state.astg == 0
        state = X.json()
        X2 = Xopt.model_validate(json.loads(state))
        X2.step()

        # test where we start simplex after random evals and then try to add a random evaluation in the middle
        X = Xopt.from_yaml(YAML)
        X.random_evaluate(3)
        X.step()
        assert X.generator._initial_simplex is not None
        assert X.generator.current_state.astg == 0
        X.step()
        assert X.generator.current_state.astg > 0
        with pytest.raises(SeqGeneratorError):
            X.random_evaluate(1)
        X.step()
        state = X.json()
        X2 = Xopt.model_validate(json.loads(state))
        X2.step()

    def test_simplex_options(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.constraints = {}
        gen = NelderMeadGenerator(vocs=test_vocs)

        with pytest.raises(ValidationError):
            gen.initial_point = {"x1": None, "x2": 0}

        with pytest.raises(ValidationError):
            gen.initial_simplex = {
                "x1": [0, 1],
                "x2": 0,
            }

    def test_simplex_agreement(self):
        """Compare between Vanilla Simplex and Xopt Simplex in full auto run mode"""

        # Scipy Simplex
        result = minimize(
            rosenbrock, [-1, -1], method="Nelder-Mead", options={"adaptive": True}
        )
        result = result.x

        # Xopt Simplex
        YAML = """
        stopping_condition:
            name: MaxEvaluationsCondition
            max_evaluations: 1000
        generator:
            name: neldermead
            initial_point: {x0: -1, x1: -1}
            adaptive: true
        evaluator:
            function: xopt.resources.test_functions.rosenbrock.evaluate_rosenbrock
        vocs:
            variables:
                x0: [-5, 5]
                x1: [-5, 5]
            objectives: {y: MINIMIZE}
        """
        X = Xopt.from_yaml(YAML)
        X.run()

        # Results should be the same
        xbest = X.data.iloc[X.data["y"].argmin()]
        assert np.isclose(xbest["x0"], result[0]) and np.isclose(
            xbest["x1"], result[1]
        ), "Xopt Simplex does not match the vanilla one"
