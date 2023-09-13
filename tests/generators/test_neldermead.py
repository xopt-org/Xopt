import pytest
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
