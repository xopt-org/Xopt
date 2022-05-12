import torch
from botorch.sampling import SobolQMCNormalSampler

from xopt.generators.bayesian import BayesianExplorationGenerator
from xopt.resources.testing import TEST_VOCS_DATA, TEST_VOCS_BASE, xtest_callable
from xopt import Xopt, Evaluator


class TestBayesianExplorationGenerator:
    def test_init(self):
        gen = BayesianExplorationGenerator(TEST_VOCS_BASE)

    def test_generate(self):
        gen = BayesianExplorationGenerator(
            TEST_VOCS_BASE,
        )
        gen.options.optim.raw_samples = 1
        gen.options.optim.num_restarts = 1
        gen.options.acq.monte_carlo_samples = 1
        gen.data = TEST_VOCS_DATA

        #candidate = gen.generate(5)
        #assert len(candidate) == 5

    def test_in_xopt(self):
        evaluator = Evaluator(xtest_callable)
        gen = BayesianExplorationGenerator(
            TEST_VOCS_BASE,
        )
        gen.options.optim.raw_samples = 1
        gen.options.optim.num_restarts = 1
        gen.options.acq.monte_carlo_samples = 1
        gen.data = TEST_VOCS_DATA

        xopt = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)

        # initialize with single initial candidate
        xopt.step()

        # now use bayes opt
        for _ in range(3):
            xopt.step()

    def test_in_xopt_w_proximal(self):
        evaluator = Evaluator(xtest_callable)
        gen = BayesianExplorationGenerator(
            TEST_VOCS_BASE,
        )
        gen.options.optim.raw_samples = 1
        gen.options.optim.num_restarts = 1
        gen.options.acq.monte_carlo_samples = 1
        gen.options.acq.proximal_lengthscales = [1.0, 1.0]
        gen.data = TEST_VOCS_DATA

        xopt = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)

        # initialize with single initial candidate
        xopt.step()

        # now use bayes opt
        for _ in range(5):
            xopt.step()

        evaluator = Evaluator(xtest_callable)
        gen = BayesianExplorationGenerator(
            TEST_VOCS_BASE,
        )
        gen.options.optim.raw_samples = 5
        gen.options.optim.num_restarts = 1
        gen.options.acq.monte_carlo_samples = 1
        gen.options.acq.proximal_lengthscales = [1.5, 1.5]
        gen.data = TEST_VOCS_DATA

        xopt = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)

        # initialize with single initial candidate
        xopt.step()

        # now use bayes opt
        for _ in range(3):
            xopt.step()
