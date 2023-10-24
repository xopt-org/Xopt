import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError
from scipy.optimize import fmin

from xopt import Xopt
from xopt.generators.scipy.neldermead import NelderMeadGenerator
from xopt.resources.test_functions.rosenbrock import rosenbrock
from xopt.resources.testing import TEST_VOCS_BASE


class TestNelderMeadGenerator:
    def test_simplex_generate_multiple_points(self):
        gen = NelderMeadGenerator(vocs=TEST_VOCS_BASE)

        # Try to generate multiple samples
        with pytest.raises(NotImplementedError):
            gen.generate(2)

    def test_simplex_options(self):
        gen = NelderMeadGenerator(vocs=TEST_VOCS_BASE)

        with pytest.raises(ValidationError):
            gen.initial_point = {"x1": None, "x2": 0}

        with pytest.raises(ValidationError):
            gen.initial_simplex = {
                "x1": [0, 1],
                "x2": 0,
            }

        with pytest.raises(ValidationError):
            gen.xatol = None

        with pytest.raises(ValidationError):
            gen.fatol = None

        gen.xatol = 1e-3
        gen.fatol = 1e-3
        assert gen.xatol == 1e-3
        assert gen.fatol == 1e-3

    def test_simplex_agreement(self):
        """Compare between Vanilla Simplex and Xopt Simplex"""

        # Scipy Simplex
        result = fmin(rosenbrock, [-1, -1])

        # Xopt Simplex
        YAML = """
        generator:
            name: neldermead
            initial_point: {x0: -1, x1: -1}
            adaptive: true
            xatol: 0.0001
            fatol: 0.0001
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
        assert (
                xbest["x0"] == result[0] and xbest["x1"] == result[1]
        ), "Xopt Simplex does not match the vanilla one"

    def test_fresh_start(self):
        inputs = []

        def wrap(x):
            inputs.append(x)
            return rosenbrock(x)

        result = fmin(wrap, [-1, -1])
        scipy_data = np.array(inputs)

        YAML = """
                generator:
                    name: neldermead
                    initial_point: {x0: -1, x1: -1}
                    adaptive: true
                    xatol: 0.0001
                    fatol: 0.0001
                evaluator:
                    function: xopt.resources.test_functions.rosenbrock.evaluate_rosenbrock
                vocs:
                    variables:
                        x0: [-5, 5]
                        x1: [-5, 5]
                    objectives: {y: MINIMIZE}
                """
        X = Xopt.from_yaml(YAML)
        for i in range(scipy_data.shape[0]):
            X.step()
            print('====================')

        data = X.vocs.variable_data(X.data).to_numpy()
        assert np.array_equal(data, scipy_data)

        xbest = X.data.iloc[X.data["y"].argmin()]
        assert (
                xbest["x0"] == result[0] and xbest["x1"] == result[1]
        ), "Xopt Simplex does not match the vanilla one"

    def test_resume_consistency(self):
        inputs = []

        def wrap(x):
            inputs.append(x)
            return rosenbrock(x)

        result = fmin(wrap, [-1, -1])
        scipy_data = np.array(inputs)

        YAML = """
                generator:
                    name: neldermead
                    initial_point: {x0: -1, x1: -1}
                    adaptive: true
                    xatol: 0.0001
                    fatol: 0.0001
                evaluator:
                    function: xopt.resources.test_functions.rosenbrock.evaluate_rosenbrock
                vocs:
                    variables:
                        x0: [-5, 5]
                        x1: [-5, 5]
                    objectives: {y: MINIMIZE}
                """
        X = Xopt.from_yaml(YAML)
        X.step()

        def compare(X, X2):
            y = yaml.safe_load(X.yaml())
            y2 = yaml.safe_load(X2.yaml())
            y.pop('data')
            y2.pop('data')
            assert y == y2
            # For unclear reasons, column order changes on reload....
            data = X.data.drop(['xopt_runtime', 'xopt_error'], axis=1)
            data2 = X2.data.drop(['xopt_runtime', 'xopt_error'], axis=1)
            # On reload, index is not a range index anymore!
            pd.testing.assert_frame_equal(data, data2, check_index_type=False)

        for i in range(scipy_data.shape[0]-1):
            state = X.yaml()
            X2 = Xopt.from_yaml(state)
            compare(X, X2)

            samples = X.generator.generate(1)
            samples2 = X2.generator.generate(1)
            compare(X, X2)
            print('>>>>>>>>>>>>>>>')

            state = X.yaml()
            X3 = Xopt.from_yaml(state)
            compare(X, X3)
            print('>>>>>>>>>>>>>>>')

            X.evaluate_data(samples)
            X2.evaluate_data(samples2)
            X3.evaluate_data(samples2)
            compare(X, X2)
            compare(X, X3)
            print('====================')

        data = X.vocs.variable_data(X.data).to_numpy()
        assert data.shape == scipy_data.shape
        assert np.array_equal(data, scipy_data)

        data = X2.vocs.variable_data(X2.data).to_numpy()
        assert data.shape == scipy_data.shape
        # Numerical precision issues with using strings for floats
        assert np.allclose(data, scipy_data, rtol=0, atol=1e-10)

        # Results should be the same
        xbest = X.data.iloc[X.data["y"].argmin()]
        assert (
                xbest["x0"] == result[0] and xbest["x1"] == result[1]
        ), "Xopt Simplex does not match the vanilla one"
