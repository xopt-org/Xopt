import math
from abc import ABC
from copy import copy, deepcopy
from concurrent.futures import ThreadPoolExecutor


import pandas as pd
import pytest
import yaml

from xopt.evaluator import Evaluator
from xopt.base import Xopt, XoptOptions
from xopt.vocs import VOCS
from xopt.errors import XoptError
from xopt.generator import Generator
from xopt.generators.random import RandomGenerator
from xopt.resources.testing import TEST_VOCS_BASE, TEST_YAML, xtest_callable


class DummyGenerator(Generator, ABC):
    def add_data(self, new_data: pd.DataFrame):
        self.data = pd.concat([self.data, new_data], axis=0)

    def generate(self, n_candidates) -> pd.DataFrame:
        pass

    def default_options(self):
        pass


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

        evaluator = Evaluator(function=dummy)
        gen = RandomGenerator(deepcopy(TEST_VOCS_BASE))
        X = Xopt(generator=gen, evaluator=evaluator, vocs=deepcopy(TEST_VOCS_BASE))

    def test_function_checking(self):
        def f(x, a=True):
            if a:
                return {"f": x["x1"] ** 2 + x["x2"] ** 2}
            else:
                return {"f": False}

        def g(x, a=True):
            return False

        vocs = VOCS(
            variables={"x": [0, 2 * math.pi]},
            objectives={"f": "MINIMIZE"},
        )

        # init with generator and evaluator
        evaluator = Evaluator(function=f)
        generator = RandomGenerator(vocs)
        X = Xopt(
            generator=generator,
            evaluator=evaluator,
            vocs=vocs,
            options=XoptOptions(strict=True),
        )
        with pytest.raises(KeyError):
            X.step()

        # init with generator and evaluator
        evaluator = Evaluator(function=g)
        generator = RandomGenerator(vocs)
        X2 = Xopt(
            generator=generator,
            evaluator=evaluator,
            vocs=vocs,
            options=XoptOptions(strict=True),
        )
        with pytest.raises(XoptError):
            X2.step()

    def test_update_data(self):
        generator = DummyGenerator(deepcopy(TEST_VOCS_BASE))
        evaluator = Evaluator(function=xtest_callable)
        X = Xopt(
            generator=generator,
            evaluator=evaluator,
            vocs=deepcopy(TEST_VOCS_BASE),
        )
        X.submit_data(pd.DataFrame({"x1": [0.0, 1.0], "x2": [0.0, 1.0]}))

        assert len(X.generator.data) == 2

    def test_asynch(self):
        evaluator = Evaluator(function=xtest_callable)
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
                function=xtest_callable, executor=ThreadPoolExecutor(), max_workers=mw
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

            # TODO: better async test. This is unpredictable:
            # assert len(X2.data) == n_steps

    def test_strict(self):
        def bad_function(inval):
            raise ValueError

        evaluator = Evaluator(function=bad_function)
        gen = RandomGenerator(deepcopy(TEST_VOCS_BASE))
        X = Xopt(generator=gen, evaluator=evaluator, vocs=deepcopy(TEST_VOCS_BASE))

        # should be able to run with strict=False (default)
        X.step()

        X2 = Xopt(generator=gen, evaluator=evaluator, vocs=deepcopy(TEST_VOCS_BASE))
        X2.options.strict = True

        with pytest.raises(ValueError):
            X2.step()

    def test_random(self):
        evaluator = Evaluator(function=xtest_callable)
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
        evaluator = Evaluator(function=xtest_callable)
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
