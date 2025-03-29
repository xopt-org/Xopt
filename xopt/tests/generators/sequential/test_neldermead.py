from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from scipy.optimize import minimize

from xopt import VOCS, Xopt
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
        X.max_evaluations = 1000
        X.run()

        # Results should be the same
        xbest = X.data.iloc[X.data["y"].argmin()]
        assert np.isclose(xbest["x0"], result[0]) and np.isclose(
            xbest["x1"], result[1]
        ), "Xopt Simplex does not match the vanilla one"

    @pytest.mark.parametrize(
        "fun, obj", [(eval_f_linear_pos, "MINIMIZE"), (eval_f_linear_neg, "MAXIMIZE")]
    )
    def test_simplex_convergence(self, fun, obj):
        variables = {f"x{i}": [-5, 5] for i in range(10)}
        objectives = {"y1": obj}
        vocs = VOCS(variables=variables, objectives=objectives)

        config = {
            "generator": {
                "name": "neldermead",
                "initial_point": {f"x{i}": 3.5 for i in range(len(variables))},
            },
            "evaluator": {"function": fun},
            "vocs": vocs,
        }
        X = Xopt.from_dict(config)
        for i in range(1000):
            X.step()

        idx, best, _ = X.vocs.select_best(X.data)
        xbest = X.vocs.variable_data(X.data.loc[idx, :]).to_numpy().flatten()
        assert np.allclose(xbest, np.zeros(10), rtol=0, atol=1e-4)
        if obj == "MINIMIZE":
            assert best[0] >= 0.0
            assert best[0] <= 0.001
        else:
            assert best[0] <= 0.0
            assert best[0] >= -0.001

    @pytest.mark.parametrize(
        "fun,fstring,x0,v",
        [
            (rosenbrock, "rosenbrock.evaluate_rosenbrock", [-1, -1], rbvocs),
            (ackley, "ackley_20.evaluate_ackley_np", [4] * 20, ackleyvocs),
        ],
    )
    def test_simplex_agreement_steps(self, fun, fstring, x0, v):
        """Compare between Vanilla Simplex and Xopt Simplex step by step"""

        inputs = []

        def wrap(x):
            inputs.append(x)
            return fun(x)

        result = minimize(
            wrap, np.array(x0), method="Nelder-Mead", options={"adaptive": True}
        )
        scipy_data = np.array(inputs)
        print(f"Have {scipy_data.shape[0]} steps")

        config = {
            "generator": {
                "name": "neldermead",
                "initial_point": {f"x{i}": x0[i] for i in range(len(x0))},
                "adaptive": True,
            },
            "evaluator": {"function": f"xopt.resources.test_functions.{fstring}"},
            "vocs": v.model_dump(),
        }
        X = Xopt.from_dict(config)
        for i in range(scipy_data.shape[0]):
            X.step()
            data = X.vocs.variable_data(X.data).to_numpy()
            if not np.array_equal(data, scipy_data[: i + 1, :]):
                raise Exception

        data = X.vocs.variable_data(X.data).to_numpy()
        assert np.array_equal(data, scipy_data)

        idx, best, _ = X.vocs.select_best(X.data)
        xbest = X.vocs.variable_data(X.data.loc[idx, :]).to_numpy().flatten()
        assert np.array_equal(xbest, result.x), (
            "Xopt Simplex does not match the vanilla one"
        )

    @pytest.mark.parametrize(
        "fun,fstring,x0,v,cstep",
        [
            (rosenbrock, "rosenbrock.evaluate_rosenbrock", [-1, -1], rbvocs, 10),
            (ackley, "ackley_20.evaluate_ackley_np", [4] * 20, ackleyvocs, 200),
        ],
    )
    def test_resume_consistency(self, fun, fstring, x0, v, cstep):
        """Compare between Vanilla Simplex and Xopt Simplex while deserializing at every step"""
        inputs = []

        def wrap(x):
            inputs.append(x)
            return fun(x)

        result = minimize(
            wrap, np.array(x0), method="Nelder-Mead", options={"adaptive": True}
        )
        scipy_data = np.array(inputs)

        config = {
            "generator": {
                "name": "neldermead",
                "initial_point": {f"x{i}": x0[i] for i in range(len(x0))},
                "adaptive": True,
            },
            "evaluator": {"function": f"xopt.resources.test_functions.{fstring}"},
            "vocs": v.model_dump(),
        }
        X = Xopt.from_dict(config)
        X.step()

        for i in range(scipy_data.shape[0] - 1):
            # For performance, only check some steps
            if i % cstep == 0 or i == scipy_data.shape[0] - 2:
                state = X.json()
                X2 = Xopt.model_validate(json.loads(state))
                compare(X, X2)

                samples = X.generator.generate(1)
                samples2 = X2.generator.generate(1)
                assert samples == samples2
                compare(X, X2)

                state = X.json()
                X3 = Xopt.model_validate(json.loads(state))
                # TODO: maybe store in dump?
                X3.generator._last_candidate = X2.generator._last_candidate
                compare(X, X3)

                X.evaluate_data(samples)
                X2.evaluate_data(samples2)
                X3.evaluate_data(samples2)
                compare(X, X2)
                compare(X, X3)
            else:
                samples = X.generator.generate(1)
                X.evaluate_data(samples)

        data = X.vocs.variable_data(X.data).to_numpy()
        assert data.shape == scipy_data.shape
        assert np.array_equal(data, scipy_data)

        data = X2.vocs.variable_data(X2.data).to_numpy()
        assert data.shape == scipy_data.shape
        # Numerical precision issues with using strings for floats, need tolerance
        assert np.allclose(data, scipy_data, rtol=0, atol=1e-10)

        data = X3.vocs.variable_data(X3.data).to_numpy()
        assert data.shape == scipy_data.shape
        assert np.allclose(data, scipy_data, rtol=0, atol=1e-10)

        idx, best, _ = X.vocs.select_best(X.data)
        xbest = X.vocs.variable_data(X.data.loc[idx, :]).to_numpy().flatten()
        assert np.array_equal(xbest, result.x), (
            "Xopt Simplex does not match the vanilla one"
        )
