from copy import deepcopy

import numpy as np
import pandas as pd

from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.mobo import MOBOGenerator
from xopt.resources.test_functions.tnk import evaluate_TNK, tnk_vocs
from xopt.resources.testing import TEST_VOCS_BASE


class TestMOBOGenerator:
    def test_init(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.objectives.update({"y2": "MINIMIZE"})
        reference_point = {"y1": 1.5, "y2": 1.5}

        MOBOGenerator(vocs=vocs, reference_point=reference_point)

    def test_script(self):
        evaluator = Evaluator(function=evaluate_TNK)
        reference_point = {"y1": 1.5, "y2": 1.5}

        gen = MOBOGenerator(vocs=tnk_vocs, reference_point=reference_point)
        print(gen.model_dump())
        gen = deepcopy(gen)
        gen.n_monte_carlo_samples = 20

        for ele in [gen]:
            dump = ele.model_dump()
            print(dump)
            generator = MOBOGenerator(vocs=tnk_vocs, **dump)
            X = Xopt(generator=generator, evaluator=evaluator, vocs=tnk_vocs)
            X.random_evaluate(3)
            X.step()

    def test_hypervolume_calculation(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.objectives.update({"y2": "MINIMIZE"})
        vocs.constraints = {}

        data = pd.DataFrame(
            {
                "x1": np.random.rand(2),
                "x2": np.random.rand(2),
                "y1": np.array((1.0, 0.0)),
                "y2": np.array((0.0, 2.0)),
            }
        )
        reference_point = {"y1": 10.0, "y2": 1.0}
        gen = MOBOGenerator(vocs=vocs, reference_point=reference_point)
        gen.add_data(data)

        assert gen.calculate_hypervolume() == 9.0

        vocs.objectives["y1"] = "MAXIMIZE"
        gen = MOBOGenerator(vocs=vocs, reference_point=reference_point)
        gen.add_data(data)

        assert gen.calculate_hypervolume() == 0.0
