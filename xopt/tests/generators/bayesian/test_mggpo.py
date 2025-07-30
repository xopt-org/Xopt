from copy import deepcopy

import pandas as pd
import pytest
import torch

from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.mggpo import MGGPOGenerator
from xopt.resources.test_functions.tnk import evaluate_TNK, tnk_vocs
from xopt.resources.testing import (
    TEST_VOCS_BASE,
    TEST_VOCS_BASE_MO,
    TEST_VOCS_BASE_MO_NC,
    TEST_VOCS_DATA_MO,
    TEST_VOCS_REF_POINT,
    check_generator_tensor_locations,
    create_set_options_helper,
)

cuda_combinations = [False] if not torch.cuda.is_available() else [False, True]
device_map = {False: torch.device("cpu"), True: torch.device("cuda:0")}


set_options = create_set_options_helper(data=TEST_VOCS_DATA_MO)


class TestMGGPO:
    def test_init(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.objectives.update({"y2": "MINIMIZE"})
        reference_point = {"y1": 3.14, "y2": 3.14}
        MGGPOGenerator(vocs=vocs, reference_point=reference_point)

    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_generate(self, use_cuda):
        gen = MGGPOGenerator(
            vocs=TEST_VOCS_BASE_MO, reference_point=TEST_VOCS_REF_POINT
        )
        set_options(gen, use_cuda, add_data=True)

        candidate = gen.generate(1)
        assert len(candidate) == 1

        candidate = gen.generate(2)
        assert len(candidate) == 2

        check_generator_tensor_locations(gen, device_map[use_cuda])

        gen = MGGPOGenerator(
            vocs=TEST_VOCS_BASE_MO_NC, reference_point=TEST_VOCS_REF_POINT
        )
        set_options(gen, use_cuda, add_data=True)

        candidate = gen.generate(1)
        assert len(candidate) == 1

        candidate = gen.generate(2)
        assert len(candidate) == 2

        check_generator_tensor_locations(gen, device_map[use_cuda])

    def test_serial(self):
        evaluator = Evaluator(function=evaluate_TNK)

        # test check options
        vocs = deepcopy(tnk_vocs)
        reference_point = {"y1": 3.14, "y2": 3.14}
        gen = MGGPOGenerator(vocs=vocs, reference_point=reference_point)
        X = Xopt(evaluator=evaluator, generator=gen, vocs=vocs)
        X.evaluate_data(pd.DataFrame({"x1": [1.0, 0.75], "x2": [0.75, 1.0]}))
        samples = X.generator.generate(10)
        assert pd.DataFrame(samples).to_numpy().shape == (10, 2)

        X.step()

    def test_bactched(self):
        evaluator = Evaluator(function=evaluate_TNK)
        evaluator.max_workers = 10

        # test check options
        vocs = deepcopy(tnk_vocs)
        reference_point = {"y1": 3.14, "y2": 3.14}
        gen = MGGPOGenerator(vocs=vocs, reference_point=reference_point)

        X = Xopt(evaluator=evaluator, generator=gen, vocs=vocs)
        X.evaluate_data(pd.DataFrame({"x1": [1.0, 0.75], "x2": [0.75, 1.0]}))

        for _ in [0, 1]:
            X.step()
