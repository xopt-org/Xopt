from xopt.evaluator import Evaluator
from xopt.base import Xopt

from xopt.generators.bayesian.bayesian_exploration import BayesianExplorationGenerator
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA, xtest_callable


class TestBayesianExplorationGenerator:
    def test_init(self):
        BayesianExplorationGenerator(TEST_VOCS_BASE)

    def test_generate(self):
        gen = BayesianExplorationGenerator(
            TEST_VOCS_BASE,
        )
        gen.options.optim.raw_samples = 1
        gen.options.optim.num_restarts = 1
        gen.options.acq.monte_carlo_samples = 1
        gen.data = TEST_VOCS_DATA

        # candidate = gen.generate(5)
        # assert len(candidate) == 5

    def test_in_xopt(self):
        evaluator = Evaluator(function=xtest_callable)
        gen = BayesianExplorationGenerator(
            TEST_VOCS_BASE,
        )
        gen.options.optim.raw_samples = 1
        gen.options.optim.num_restarts = 1
        gen.options.acq.monte_carlo_samples = 1
        gen.data = TEST_VOCS_DATA

        X = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)

        # initialize with single initial candidate
        X.step()

        # now use bayes opt
        X.step()

    def test_in_xopt_w_proximal(self):
        evaluator = Evaluator(function=xtest_callable)
        gen = BayesianExplorationGenerator(
            TEST_VOCS_BASE,
        )
        gen.options.optim.raw_samples = 1
        gen.options.optim.num_restarts = 1
        gen.options.acq.monte_carlo_samples = 1
        gen.options.acq.proximal_lengthscales = [1.0, 1.0]
        gen.data = TEST_VOCS_DATA

        X = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)

        # initialize with single initial candidate
        X.step()

        # now use bayes opt
        X.step()

        evaluator = Evaluator(function=xtest_callable)
        gen = BayesianExplorationGenerator(
            TEST_VOCS_BASE,
        )
        gen.options.optim.raw_samples = 2
        gen.options.optim.num_restarts = 1
        gen.options.acq.monte_carlo_samples = 1
        gen.options.acq.proximal_lengthscales = [1.5, 1.5]
        gen.data = TEST_VOCS_DATA

        X = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)

        # initialize with single initial candidate
        X.step()
        X.step()
