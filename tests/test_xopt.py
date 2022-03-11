from xopt import XoptBase, Evaluator
from xopt.generators.random import RandomGenerator
from xopt.resources.testing import TEST_VOCS_BASE, test_callable


class TestXopt:
    def test_init(self):
        pass

    def test_random(self):
        evaluator = Evaluator(test_callable)
        generator = RandomGenerator(TEST_VOCS_BASE)

        xopt = XoptBase(generator, evaluator, TEST_VOCS_BASE)
        xopt.step()
        assert set(xopt.history.keys()) == {*TEST_VOCS_BASE.variables,
                                            *TEST_VOCS_BASE.objectives,
                                            *TEST_VOCS_BASE.constraints, "done"
                                            }

        for _ in range(10):
            xopt.step()
        history = xopt.history
        assert len(history) == 11
