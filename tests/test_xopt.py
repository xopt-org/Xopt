import pytest

from xopt import XoptBase, Evaluator
from xopt.generators.random import RandomGenerator
from xopt.resources.testing import TEST_VOCS_BASE, xtest_callable


def bad_function(input):
    raise ValueError

class TestXopt:
    def test_init(self):
        pass

    def test_strict(self):
        eval = Evaluator(bad_function)
        gen = RandomGenerator(TEST_VOCS_BASE)
        X = XoptBase(generator=gen, evaluator=eval, vocs=TEST_VOCS_BASE)

        # should be able to run with strict=False (default)
        X.step()

        X2 = XoptBase(generator=gen, evaluator=eval, vocs=TEST_VOCS_BASE, strict=True)

        with pytest.raises(ValueError):
            X2.step()

    def test_random(self):
        evaluator = Evaluator(xtest_callable)
        generator = RandomGenerator(TEST_VOCS_BASE)

        xopt = XoptBase(generator=generator, evaluator=evaluator, vocs=TEST_VOCS_BASE)
        xopt.step()

        for _ in range(10):
            xopt.step()
        data = xopt.data
        assert len(data) == 11
