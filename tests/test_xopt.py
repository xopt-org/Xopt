from copy import copy, deepcopy

import pytest
import yaml

from xopt import Evaluator, Xopt
from xopt.errors import XoptError
from xopt.generators.random import RandomGenerator
from xopt.options import XoptOptions
from xopt.resources.testing import TEST_VOCS_BASE, TEST_YAML, xtest_callable


class TestXopt:
    def test_init(self):
        # init with no arguments
        with pytest.raises(XoptError):
            X = Xopt()

        # init with YAML
        X = Xopt(config=yaml.safe_load(copy(TEST_YAML)))
        assert X.evaluator.function == xtest_callable
        assert isinstance(X.generator, RandomGenerator)

        # init with generator and evaluator
        def dummy(x):
            pass

        evaluator = Evaluator(dummy)
        gen = RandomGenerator(deepcopy(TEST_VOCS_BASE))
        X = Xopt(generator=gen, evaluator=evaluator, vocs=deepcopy(TEST_VOCS_BASE))

    def test_asynch(self):
        evaluator = Evaluator(xtest_callable)
        generator = RandomGenerator(deepcopy(TEST_VOCS_BASE))
        X = Xopt(
            generator=generator,
            evaluator=evaluator,
            vocs=deepcopy(TEST_VOCS_BASE),
            options=XoptOptions(asynch=True),
        )
        n_steps = 5
        for i in range(n_steps):
            X.step()
        assert len(X.data) == n_steps

        # now use a threadpool evaluator with different number of max workers
        for mw in [2]:
            evaluator = Evaluator(
                xtest_callable, executor="ThreadPoolExecutor", max_workers=mw
            )
            X2 = Xopt(
                generator=generator,
                evaluator=evaluator,
                vocs=deepcopy(TEST_VOCS_BASE),
                options=XoptOptions(asynch=True),
            )

            n_steps = 5
            for i in range(n_steps):
                X2.step()
            assert len(X2.data) == n_steps

    def test_strict(self):
        def bad_function(inval):
            raise ValueError

        evaluator = Evaluator(bad_function)
        gen = RandomGenerator(deepcopy(TEST_VOCS_BASE))
        X = Xopt(generator=gen, evaluator=evaluator, vocs=deepcopy(TEST_VOCS_BASE))

        # should be able to run with strict=False (default)
        X.step()

        X2 = Xopt(generator=gen, evaluator=evaluator, vocs=deepcopy(TEST_VOCS_BASE))
        X2.options.strict = True

        with pytest.raises(ValueError):
            X2.step()

    def test_random(self):
        evaluator = Evaluator(xtest_callable)
        generator = RandomGenerator(deepcopy(TEST_VOCS_BASE))

        xopt = Xopt(
            generator=generator, evaluator=evaluator, vocs=deepcopy(TEST_VOCS_BASE)
        )
        xopt.step()

        for _ in range(10):
            xopt.step()
        data = xopt.data
        assert len(data) == 11

    def test_checkpointing(self):
        evaluator = Evaluator(xtest_callable)
        generator = RandomGenerator(deepcopy(TEST_VOCS_BASE))

        X = Xopt(
            generator=generator, evaluator=evaluator, vocs=deepcopy(TEST_VOCS_BASE)
        )
        X.options.dump_file = "test_checkpointing.yaml"

        try:
            X.step()

            for _ in range(5):
                X.step()

            # try to load the state from nothing
            with open(X.options.dump_file, "r") as f:
                config = yaml.safe_load(f)
            X2 = Xopt(config=config)

            for _ in range(5):
                X2.step()

            assert len(X2.data) == 11
            assert X2._ix_last == 11

        except Exception as e:
            raise e
        finally:
            # clean up
            import os

            os.remove(X.options.dump_file)
