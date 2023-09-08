import math
import os
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy

import numpy as np
import pandas as pd
import pytest
import yaml

from xopt.base import Xopt, XoptOptions
from xopt.errors import XoptError
from xopt.evaluator import Evaluator
from xopt.generator import Generator
from xopt.generators.random import RandomGenerator
from xopt.resources.testing import TEST_VOCS_BASE, TEST_YAML, xtest_callable
from xopt.vocs import VOCS


class DummyGenerator(Generator, ABC):
    name = "dummy"

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
            Xopt()

        # init with YAML
        X = Xopt(config=yaml.safe_load(copy(TEST_YAML)))
        assert X.evaluator.function == xtest_callable
        assert isinstance(X.generator, RandomGenerator)

        # init with generator and evaluator
        def dummy(x):
            pass

        evaluator = Evaluator(function=dummy)
        gen = RandomGenerator(vocs=deepcopy(TEST_VOCS_BASE))
        Xopt(generator=gen, evaluator=evaluator, vocs=deepcopy(TEST_VOCS_BASE))

    def test_evaluate_data(self):
        evaluator = Evaluator(function=xtest_callable)
        generator = RandomGenerator(vocs=deepcopy(TEST_VOCS_BASE))

        xopt = Xopt(
                generator=generator, evaluator=evaluator, vocs=deepcopy(TEST_VOCS_BASE)
        )

        # test evaluating data w/o constants specified
        test_data = deepcopy(TEST_VOCS_BASE).random_inputs(3)
        # pop constant specified in vocs
        test_data.pop("cnt1")
        xopt.evaluate_data(pd.DataFrame(test_data))

        assert np.all(xopt.data["cnt1"].to_numpy() == np.ones(3))

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
        generator = RandomGenerator(vocs=vocs)
        X = Xopt(
                generator=generator,
                evaluator=evaluator,
                vocs=vocs,
                options=XoptOptions(strict=True),
        )
        with pytest.raises(XoptError):
            X.step()

        # init with generator and evaluator
        evaluator = Evaluator(function=g)  # Has non-dict return type
        generator = RandomGenerator(vocs=vocs)
        X2 = Xopt(
                generator=generator,
                evaluator=evaluator,
                vocs=vocs,
                options=XoptOptions(strict=True),
        )
        with pytest.raises(XoptError):
            X2.step()

    def test_submit_bad_data(self):
        generator = DummyGenerator(vocs=deepcopy(TEST_VOCS_BASE))
        evaluator = Evaluator(function=xtest_callable)
        X = Xopt(
                generator=generator,
                evaluator=evaluator,
                vocs=deepcopy(TEST_VOCS_BASE),
        )
        with pytest.raises(ValueError):
            X.evaluate_data(pd.DataFrame({"x1": [0.0, 5.0], "x2": [-3.0, 1.0]}))

    def test_add_data(self):
        generator = DummyGenerator(vocs=deepcopy(TEST_VOCS_BASE))
        evaluator = Evaluator(function=xtest_callable)
        X = Xopt(
                generator=generator,
                evaluator=evaluator,
                vocs=deepcopy(TEST_VOCS_BASE),
        )
        assert len(X.generator.data) == 0
        X.add_data(pd.DataFrame({"x1": [0.0, 1.0], "x2": [0.0, 1.0]}))

        assert (
                len(X.generator.data) == 2
        ), f"len(X.generator.data) = {len(X.generator.data)}"

    def test_asynch(self):
        evaluator = Evaluator(function=xtest_callable)
        generator = RandomGenerator(vocs=deepcopy(TEST_VOCS_BASE))
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
        gen = RandomGenerator(vocs=deepcopy(TEST_VOCS_BASE))
        X = Xopt(generator=gen, evaluator=evaluator, vocs=deepcopy(TEST_VOCS_BASE))

        # should raise an error (default)
        with pytest.raises(XoptError):
            X.step()

        X2 = Xopt(generator=gen, evaluator=evaluator, vocs=deepcopy(TEST_VOCS_BASE))
        X2.options.strict = False

        X2.random_evaluate(10)
        # should run fine
        X2.step()
        assert "xopt_error_str" in X2.data.columns

    def test_process_futures(self):
        ss = 0

        def bad_function_sometimes(inval):
            if ss:
                raise ValueError
            else:
                return {"y1": 0.0, "c1": 0.0}

        evaluator = Evaluator(function=bad_function_sometimes)
        gen = RandomGenerator(vocs=deepcopy(TEST_VOCS_BASE))
        X = Xopt(generator=gen, evaluator=evaluator, vocs=deepcopy(TEST_VOCS_BASE))
        X.options.strict = False

        # Submit to the evaluator some new inputs
        X.submit_data(deepcopy(TEST_VOCS_BASE).random_inputs(4))
        X.process_futures()

        ss = 1
        X.submit_data(deepcopy(TEST_VOCS_BASE).random_inputs(4))
        X.process_futures()

    def test_checkpointing(self):
        evaluator = Evaluator(function=xtest_callable)
        generator = RandomGenerator(vocs=deepcopy(TEST_VOCS_BASE))

        X = Xopt(
                generator=generator, evaluator=evaluator, vocs=deepcopy(TEST_VOCS_BASE)
        )
        X.options.dump_file = "test_checkpointing.yaml"

        X.step()

        for _ in range(5):
            X.step()

        # try to load the state from nothing
        with open(X.options.dump_file, "r") as f:
            config = yaml.safe_load(f)
        os.remove(X.options.dump_file)
        X2 = Xopt(config=config)

        for _ in range(5):
            X2.step()

        assert len(X2.data) == 11
        assert X2._ix_last == 11

    @pytest.fixture(scope='module', autouse=True)
    def clean_up(self):
        yield
        files = ['test_checkpointing.yaml']
        for f in files:
            if os.path.exists(f):
                os.remove(f)

    def test_random_evaluate(self):
        evaluator = Evaluator(function=xtest_callable)
        generator = RandomGenerator(vocs=deepcopy(TEST_VOCS_BASE))

        xopt = Xopt(
                generator=generator, evaluator=evaluator, vocs=deepcopy(TEST_VOCS_BASE)
        )
        xopt.random_evaluate(2)
        xopt.random_evaluate(1)
        assert len(xopt.data) == 3
