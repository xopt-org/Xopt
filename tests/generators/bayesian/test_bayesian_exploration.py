import torch
from botorch.sampling import SobolQMCNormalSampler

from xopt.generators.bayesian import BayesianExplorationGenerator
from xopt.resources.testing import TEST_VOCS_DATA, TEST_VOCS_BASE, test_callable
from xopt import XoptBase, Evaluator


class TestBayesianExplorationGenerator:
    def test_init(self):
        gen = BayesianExplorationGenerator(TEST_VOCS_BASE)

    def test_generate(self):
        gen = BayesianExplorationGenerator(
            TEST_VOCS_BASE,
            raw_samples=1,
            num_restarts=1,
            sampler=SobolQMCNormalSampler(1)
        )

        gen.data = TEST_VOCS_DATA
        candidate = gen.generate(5)
        assert len(candidate) == 5

    def test_in_xopt(self):
        evaluator = Evaluator(test_callable)
        generator = BayesianExplorationGenerator(
            TEST_VOCS_BASE,
            raw_samples=1,
            num_restarts=1,
            sampler=SobolQMCNormalSampler(1)
        )

        xopt = XoptBase(generator, evaluator, TEST_VOCS_BASE)

        # initialize with single initial candidate
        xopt.step()

        # now use bayes opt
        for _ in range(1):
            xopt.step()

    def test_in_xopt_w_proximal(self):
        evaluator = Evaluator(test_callable)
        generator = BayesianExplorationGenerator(
            TEST_VOCS_BASE,
            raw_samples=1,
            num_restarts=1,
            sampler=SobolQMCNormalSampler(1),
            proximal_lengthscales=[1.0, 1.0]
        )

        xopt = XoptBase(generator, evaluator, TEST_VOCS_BASE)

        # initialize with single initial candidate
        xopt.step()

        # now use bayes opt
        for _ in range(1):
            xopt.step()
