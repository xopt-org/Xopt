from copy import deepcopy

import pytest

from pydantic import ValidationError

from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.bayesian_exploration import BayesianExplorationGenerator
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA, xtest_callable


class TestBayesianExplorationGenerator:
    def test_generate(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs2 = deepcopy(test_vocs)
        test_vocs2.objectives = {}
        test_vocs2.observables = ["y1"]

        for ele in [test_vocs2]:
            gen = BayesianExplorationGenerator(
                vocs=ele,
            )
            gen.numerical_optimizer.n_restarts = 1
            gen.n_monte_carlo_samples = 1
            gen.data = TEST_VOCS_DATA

            candidate = gen.generate(1)
            assert len(candidate) == 1
            candidate = gen.generate(5)
            assert len(candidate) == 5

            # test without constraints
            gen = BayesianExplorationGenerator(
                vocs=ele,
            )
            gen.numerical_optimizer.n_restarts = 1
            gen.n_monte_carlo_samples = 1
            gen.data = TEST_VOCS_DATA

            candidate = gen.generate(1)
            assert len(candidate) == 1
            candidate = gen.generate(5)
            assert len(candidate) == 5

    def test_in_xopt(self):
        evaluator = Evaluator(function=xtest_callable)

        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives = {}
        test_vocs.observables = ["y1"]

        gen = BayesianExplorationGenerator(vocs=test_vocs)
        gen.numerical_optimizer.n_restarts = 1
        gen.n_monte_carlo_samples = 1
        gen.data = TEST_VOCS_DATA

        X = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)

        # now use bayes opt
        X.step()
        X.step()

    def test_interpolation(self):
        evaluator = Evaluator(function=xtest_callable)

        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives = {}
        test_vocs.observables = ["y1"]

        gen = BayesianExplorationGenerator(vocs=test_vocs)
        gen.numerical_optimizer.n_restarts = 1
        gen.n_monte_carlo_samples = 1
        gen.n_interpolate_points = 5

        X = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)
        X.add_data(TEST_VOCS_DATA)

        # now use bayes opt
        X.step()
        X.step()
        assert len(X.data) == 20

    def test_vocs_validation(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)

        with pytest.raises(ValidationError):
            BayesianExplorationGenerator(vocs=test_vocs)
