import numpy as np
import pandas as pd

from xopt import Evaluator
from xopt.evaluator import EvaluatorOptions


class TestEvaluator:
    @staticmethod
    def f(x):
        return x["x1"] ** 2 + x["x2"] ** 2

    def test_submit(self):
        evaluator = Evaluator(self.f)
        candidates = pd.DataFrame(np.random.rand(10, 2), columns=['x1', 'x2'])
        futures = evaluator.submit_data(candidates)
        assert len(futures) == 10

    def test_init(self):
        test_dict = {
            "function": self.f,
            "max_workers": 1,
        }
        evaluator = Evaluator(**test_dict)
        candidates = pd.DataFrame(np.random.rand(10, 2), columns=['x1', 'x2'])
        futures = evaluator.submit_data(candidates)
        assert len(futures) == 10

        evaluator = Evaluator.from_options(EvaluatorOptions(**test_dict))
        candidates = pd.DataFrame(np.random.rand(10, 2), columns=['x1', 'x2'])
        futures = evaluator.submit_data(candidates)
        assert len(futures) == 10

    def test_serialize(self):
        test_dict = {
            "function": self.f,
            "max_workers": 1,
        }

        options = EvaluatorOptions(**test_dict)

