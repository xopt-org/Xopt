from copy import deepcopy

import pandas as pd

from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.mggpo import MGGPOGenerator
from xopt.resources.test_functions.tnk import evaluate_TNK, tnk_vocs
from xopt.resources.testing import TEST_VOCS_BASE


class TestMGPO:
    def test_init(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.objectives.update({"y2": "MINIMIZE"})
        MGGPOGenerator(vocs)

    def test_serial(self):
        evaluator = Evaluator(function=evaluate_TNK)

        # test check options
        vocs = deepcopy(tnk_vocs)
        gen = MGGPOGenerator(vocs)
        gen.options.acq.reference_point = {"y1": 3.14, "y2": 3.14}
        X = Xopt(evaluator=evaluator, generator=gen, vocs=vocs)
        X.evaluate_data(pd.DataFrame({"x1": [1.0, 0.75], "x2": [0.75, 1.0]}))
        samples = X.generator.generate(10)
        assert samples.to_numpy().shape == (10, 3)

        X.step()

    def test_bactched(self):
        evaluator = Evaluator(function=evaluate_TNK)
        evaluator.max_workers = 10

        # test check options
        vocs = deepcopy(tnk_vocs)
        gen = MGGPOGenerator(vocs)
        gen.options.acq.reference_point = {"y1": 3.14, "y2": 3.14}

        X = Xopt(evaluator=evaluator, generator=gen, vocs=vocs)
        X.evaluate_data(pd.DataFrame({"x1": [1.0, 0.75], "x2": [0.75, 1.0]}))

        for _ in [0, 1]:
            X.step()
        print(X.data)
