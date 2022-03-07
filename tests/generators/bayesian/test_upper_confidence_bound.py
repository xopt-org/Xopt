from xopt.resources.testing import TEST_VOCS_DATA, TEST_VOCS_BASE, test_callable
from xopt import Xopt, Evaluator
from xopt.generators.bayesian.upper_confidence_bound import \
    UpperConfidenceBoundGenerator


class TestUpperConfidenceBoundGenerator:
    def test_init(self):
        ucb_gen = UpperConfidenceBoundGenerator(TEST_VOCS_BASE)

    def test_generate(self):
        ucb_gen = UpperConfidenceBoundGenerator(TEST_VOCS_BASE)
        candidate = ucb_gen.generate(TEST_VOCS_DATA, 5)

    def test_in_xopt(self):
        evaluator = Evaluator(test_callable)
        generator = UpperConfidenceBoundGenerator(TEST_VOCS_BASE)

        xopt = Xopt(generator, evaluator, TEST_VOCS_BASE)

        # initialize with single initial candidate
        xopt.step()

        # now use bayes opt
        for _ in range(1):
            xopt.step()
