from xopt import XoptBase, Evaluator
from xopt.generators.random import RandomGenerator
from xopt.resources.testing import TEST_VOCS_BASE, xtest_callable


class TestXopt:
    def test_init(self):
        pass

    def test_random(self):
        evaluator = Evaluator(xtest_callable)
        generator = RandomGenerator(TEST_VOCS_BASE)

        xopt = XoptBase(generator, evaluator, TEST_VOCS_BASE)
        xopt.step()

        for _ in range(10):
            xopt.step()
        data = xopt.data
        assert len(data) == 10
