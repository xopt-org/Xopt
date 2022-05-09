import numpy as np
import pandas as pd

from xopt import Evaluator
from xopt.evaluator import EvaluatorOptions


class TestEvaluator:
    @staticmethod
    def f(x, a=True):
        if a:
            return {"f": x["x1"] ** 2 + x["x2"] ** 2}
        else:
            return {"f": False}

    @staticmethod
    def g(x, a=True):
        return False

    def test_submit(self):
        evaluator = Evaluator(self.f)
        candidates = pd.DataFrame(np.random.rand(10, 2), columns=["x1", "x2"])
        futures = evaluator.submit_data(candidates)
        assert len(futures) == 10

        # try with a bad function
        evaluator = Evaluator(self.g)
        candidates = pd.DataFrame(np.random.rand(10, 2), columns=["x1", "x2"])
        futures = evaluator.submit_data(candidates)
        assert len(futures) == 10

    def test_init(self):
        test_dict = {
            "function": self.f,
            "max_workers": 1,
        }
        evaluator = Evaluator(**test_dict)
        candidates = pd.DataFrame(np.random.rand(10, 2), columns=["x1", "x2"])
        futures = evaluator.submit_data(candidates)
        assert len(futures) == 10

        evaluator = Evaluator.from_options(EvaluatorOptions(**test_dict))
        candidates = pd.DataFrame(np.random.rand(10, 2), columns=["x1", "x2"])
        futures = evaluator.submit_data(candidates)
        assert len(futures) == 10

        test_dict.update({"a": 2})
        evaluator = Evaluator(**test_dict)
        candidates = pd.DataFrame(np.random.rand(10, 2), columns=["x1", "x2"])
        futures = evaluator.submit_data(candidates)
        assert len(futures) == 10
        assert futures[0].result()["f"] is False

    def test_serialize(self):
        test_dict = {
            "function": self.f,
            "max_workers": 1,
        }

        for ele in ["ThreadPoolExecutor", "ProcessPoolExecutor"]:
            test_dict["executor"] = ele
            options = EvaluatorOptions(**test_dict)
            options.json()
