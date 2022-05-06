import pytest

from xopt import Xopt, Evaluator
from xopt.generators.random import RandomGenerator
from xopt.resources.testing import TEST_VOCS_BASE, xtest_callable


class TestXopt:
    def test_init(self):
        Xopt()

    def test_strict(self):
        def bad_function(inval):
            raise ValueError

        evaluator = Evaluator(bad_function)
        gen = RandomGenerator(TEST_VOCS_BASE)
        X = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)

        # should be able to run with strict=False (default)
        X.step()

        X2 = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)
        X2.options.strict = True

        with pytest.raises(ValueError):
            X2.step()

    def test_random(self):
        evaluator = Evaluator(xtest_callable)
        generator = RandomGenerator(TEST_VOCS_BASE)

        xopt = Xopt(generator=generator, evaluator=evaluator, vocs=TEST_VOCS_BASE)
        xopt.step()

        for _ in range(10):
            xopt.step()
        data = xopt.data
        assert len(data) == 11

    def test_checkpointing(self):
        evaluator = Evaluator(xtest_callable)
        generator = RandomGenerator(TEST_VOCS_BASE)

        xopt = Xopt(generator=generator, evaluator=evaluator, vocs=TEST_VOCS_BASE)
        xopt.options.dump_file = "test_checkpointing.yaml"
        xopt.step()

        for _ in range(5):
            xopt.step()
