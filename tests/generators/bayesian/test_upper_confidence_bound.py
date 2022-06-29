from xopt.evaluator import Evaluator
from xopt.base import Xopt
from xopt.generators.bayesian.upper_confidence_bound import (
    UpperConfidenceBoundGenerator,
)

from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA, xtest_callable


class TestUpperConfidenceBoundGenerator:
    def test_init(self):
        ucb_gen = UpperConfidenceBoundGenerator(TEST_VOCS_BASE)
        ucb_gen.options.dict()
        ucb_gen.options.schema()

    def test_generate(self):
        gen = UpperConfidenceBoundGenerator(
            TEST_VOCS_BASE,
        )
        gen.options.optim.raw_samples = 1
        gen.options.optim.num_restarts = 1
        gen.options.acq.monte_carlo_samples = 1
        gen.data = TEST_VOCS_DATA

        candidate = gen.generate(1)
        assert len(candidate) == 1

        # candidate = gen.generate(2)
        # assert len(candidate) == 2

    def test_in_xopt(self):
        evaluator = Evaluator(function=xtest_callable)
        ucb_gen = UpperConfidenceBoundGenerator(
            TEST_VOCS_BASE,
        )
        ucb_gen.options.optim.raw_samples = 1
        ucb_gen.options.optim.num_restarts = 1

        xopt = Xopt(generator=ucb_gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)

        # initialize with single initial candidate
        xopt.step()

        # now use bayes opt
        for _ in range(1):
            xopt.step()

    def test_in_xopt_w_proximal(self):
        evaluator = Evaluator(function=xtest_callable)
        ucb_gen = UpperConfidenceBoundGenerator(
            TEST_VOCS_BASE,
        )
        ucb_gen.options.optim.raw_samples = 1
        ucb_gen.options.optim.num_restarts = 1
        ucb_gen.options.acq.proximal_lengthscales = [1.0, 1.0]

        xopt = Xopt(generator=ucb_gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)

        # initialize with single initial candidate
        xopt.step()

        # now use bayes opt
        for _ in range(1):
            xopt.step()
