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
        reference_point = {"y1": 1.5, "y2": 1.5}

        MOBOGenerator(vocs=vocs, reference_point=reference_point)

    def test_script(self):
        evaluator = Evaluator(function=evaluate_TNK)
        reference_point = {"y1": 1.5, "y2": 1.5}

        # test check options
        options = MOBOGenerator(vocs=tnk_vocs, reference_point=reference_point)

        bad_options = deepcopy(options)
        bad_options.optimization_options.raw_samples = 5
        bad_options.acquisition_options.monte_carlo_samples = 10
        bad_options.acquisition_options.proximal_lengthscales = [1.0, 1.0, 1.0]

        with pytest.raises(ValueError):
            MOBOGenerator(**bad_options.dict())

        options = MOBOGenerator(vocs=tnk_vocs, reference_point=reference_point)
        base_options = deepcopy(options)
        base_options.acquisition_options.monte_carlo_samples = 20

        proximal_biasing = deepcopy(options)
        proximal_biasing.optimization_options.num_restarts = 1  # required
        proximal_biasing.acquisition_options.proximal_lengthscales = [1.0, 1.0]

        for ele in [base_options, proximal_biasing]:
            generator = MOBOGenerator(vocs=tnk_vocs, **ele.dict())
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
