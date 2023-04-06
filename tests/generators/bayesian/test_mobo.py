from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.mobo import MOBOGenerator
from xopt.resources.test_functions.tnk import evaluate_TNK, tnk_vocs
from xopt.resources.testing import TEST_VOCS_BASE


class TestMOBOGenerator:
    def test_init(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.objectives.update({"y2": "MINIMIZE"})
        gen = MOBOGenerator(vocs)

        print(f"\n{gen.options.dict()}")

    def test_script(self):
        evaluator = Evaluator(function=evaluate_TNK)

        # test check options
        bad_options = deepcopy(MOBOGenerator.default_options())
        bad_options.acq.proximal_lengthscales = [1.0, 1.0]

        bad_options2 = deepcopy(MOBOGenerator.default_options())
        bad_options2.optim.raw_samples = 5
        bad_options2.acq.monte_carlo_samples = 10
        bad_options2.acq.proximal_lengthscales = [1.0, 1.0, 1.0]

        for ele in [bad_options, bad_options2]:
            with pytest.raises(ValueError):
                MOBOGenerator(tnk_vocs, ele)

        base_options = deepcopy(MOBOGenerator.default_options())
        base_options.acq.reference_point = {"y1": 1.5, "y2": 1.5}
        base_options.acq.monte_carlo_samples = 20

        proximal_biasing = deepcopy(base_options)
        proximal_biasing.acq.reference_point = {"y1": 1.5, "y2": 1.5}
        proximal_biasing.optim.num_restarts = 1  # required
        proximal_biasing.acq.proximal_lengthscales = [1.0, 1.0]

        proximal_biasing2 = deepcopy(base_options)
        proximal_biasing2.acq.reference_point = {"y1": 1.5, "y2": 1.5}
        proximal_biasing2.optim.num_restarts = 1  # required
        proximal_biasing2.acq.proximal_lengthscales = np.array([1.0, 1.0])

        for ele in [base_options, proximal_biasing, proximal_biasing2]:
            generator = MOBOGenerator(tnk_vocs, ele)
            X = Xopt(generator=generator, evaluator=evaluator, vocs=tnk_vocs)
            X.step()
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

        options = MOBOGenerator.default_options()
        options.acq.reference_point = {"y1": 10.0, "y2": 1.0}

        generator = MOBOGenerator(vocs, options)
        generator.add_data(data)

        assert generator.calculate_hypervolume() == 9.0

        vocs.objectives["y1"] = "MAXIMIZE"
        generator = MOBOGenerator(vocs, options)
        generator.add_data(data)

        assert generator.calculate_hypervolume() == 0.0
