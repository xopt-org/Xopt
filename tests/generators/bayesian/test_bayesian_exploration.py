from copy import deepcopy

from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.bayesian_exploration import BayesianExplorationGenerator
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA, xtest_callable


class TestBayesianExplorationGenerator:
    def test_init(self):
        BayesianExplorationGenerator(vocs=TEST_VOCS_BASE)

    def test_generate(self):
        gen = BayesianExplorationGenerator(
            vocs=TEST_VOCS_BASE,
        )
        gen.optimization_options.raw_samples = 1
        gen.optimization_options.num_restarts = 1
        gen.acquisition_options.monte_carlo_samples = 1
        gen.data = TEST_VOCS_DATA

        candidate = gen.generate(1)
        assert len(candidate) == 1
        candidate = gen.generate(5)
        assert len(candidate) == 5

        # test without constraints
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.constraints = {}
        gen = BayesianExplorationGenerator(
            vocs=vocs,
        )
        gen.optimization_options.raw_samples = 1
        gen.optimization_options.num_restarts = 1
        gen.acquisition_options.monte_carlo_samples = 1
        gen.data = TEST_VOCS_DATA

        candidate = gen.generate(1)
        assert len(candidate) == 1
        candidate = gen.generate(5)
        assert len(candidate) == 5

    def test_in_xopt(self):
        evaluator = Evaluator(function=xtest_callable)
        gen = BayesianExplorationGenerator(vocs=TEST_VOCS_BASE)
        gen.optimization_options.raw_samples = 1
        gen.optimization_options.num_restarts = 1
        gen.acquisition_options.monte_carlo_samples = 1
        gen.data = TEST_VOCS_DATA

        X = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)

        # now use bayes opt
        X.step()
        X.step()

    def test_in_xopt_w_proximal(self):
        evaluator = Evaluator(function=xtest_callable)
        gen = BayesianExplorationGenerator(vocs=TEST_VOCS_BASE)
        gen.optimization_options.raw_samples = 1
        gen.optimization_options.num_restarts = 1
        gen.acquisition_options.monte_carlo_samples = 1
        gen.acquisition_options.proximal_lengthscales = [1.0, 1.0]
        gen.data = TEST_VOCS_DATA

        X = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)
        X.step()
