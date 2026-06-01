import pytest
from copy import deepcopy
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for tests
from xopt.base import Xopt
from xopt.generators.bayesian.bax.visualize import visualize_virtual_objective
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.bax.algorithms import GridOptimize
from xopt.generators.bayesian.bax_generator import BaxGenerator
from xopt.resources.testing import TEST_VOCS_BASE, xtest_callable


class TestVisualizeBax:
    @pytest.fixture
    def bax_generator(self):
        evaluator = Evaluator(function=xtest_callable)
        alg = GridOptimize()

        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives = {}
        test_vocs.observables = ["y1"]
        gen = BaxGenerator(
            vocs=test_vocs,
            algorithm=alg,
        )

        gen.numerical_optimizer.n_restarts = 1

        xopt = Xopt(generator=gen, evaluator=evaluator)

        xopt.random_evaluate(3)
        xopt.step()
        return xopt.generator

    def test_visualize_virtual_objective_1d(self, bax_generator):
        fig, ax = visualize_virtual_objective(bax_generator, variable_names=["x1"])
        assert hasattr(fig, "add_subplot")
        assert hasattr(ax, "plot") or isinstance(ax, list)

    def test_visualize_virtual_objective_2d(self, bax_generator):
        fig, ax = visualize_virtual_objective(
            bax_generator, variable_names=["x1", "x2"]
        )
        assert hasattr(fig, "add_subplot")
        assert isinstance(ax, (list, tuple, np.ndarray))

    def test_invalid_variable_names(self, bax_generator):
        with pytest.raises(ValueError):
            visualize_virtual_objective(bax_generator, variable_names=["bad"])

    def test_invalid_reference_point(self, bax_generator):
        with pytest.raises(ValueError):
            visualize_virtual_objective(
                bax_generator, variable_names=["x1"], reference_point={"bad": 0}
            )

    def test_missing_model(self, bax_generator):
        bax_generator.model = None
        with pytest.raises(ValueError):
            visualize_virtual_objective(bax_generator, variable_names=["x1"])

    def test_unsupported_dim_x(self, bax_generator):
        # Add a third variable to VOCS for this test
        evaluator = Evaluator(function=xtest_callable)
        alg = GridOptimize()

        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.variable_names.append("x3")
        test_vocs.variables["x3"] = test_vocs.variables["x1"]  # Copy domain
        test_vocs.objectives = {}
        test_vocs.observables = ["y1"]
        gen = BaxGenerator(
            vocs=test_vocs,
            algorithm=alg,
        )

        gen.numerical_optimizer.n_restarts = 1

        xopt = Xopt(generator=gen, evaluator=evaluator)

        # initialize with single initial candidate
        xopt.random_evaluate(3)
        with pytest.raises(ValueError):
            visualize_virtual_objective(
                bax_generator, variable_names=["x1", "x2", "x3"]
            )
