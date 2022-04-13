import numpy as np
import pandas as pd

from xopt import Evaluator


class TestEvaluator:
    def test_submit(self):
        def f(x):
            return x["x1"] ** 2 + x["x2"] ** 2

        evaluator = Evaluator(f)
        candidates = pd.DataFrame(np.random.rand(10, 2), columns=['x1', 'x2'])
        futures = evaluator.submit(candidates)
        assert len(futures) == 10

