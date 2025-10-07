import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from xopt.base import Xopt
from xopt.errors import GeneratorWarning
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.upper_confidence_bound import (
    UpperConfidenceBoundGenerator,
)
from xopt.resources.testing import (
    TEST_VOCS_BASE,
    TEST_VOCS_DATA,
    check_generator_tensor_locations,
    create_set_options_helper,
    generate_without_warnings,
    xtest_callable,
)

cuda_combinations = [False] if not torch.cuda.is_available() else [False, True]
device_map = {False: torch.device("cpu"), True: torch.device("cuda:0")}


set_options = create_set_options_helper(data=TEST_VOCS_DATA)


class TestUpperConfidenceBoundGenerator:
    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_init(self, use_cuda):
        gen = UpperConfidenceBoundGenerator(vocs=TEST_VOCS_BASE)
        gen.use_cuda = use_cuda
        gen.model_dump_json()
        check_generator_tensor_locations(gen, device_map[use_cuda])

        # test init from dict
        gen2 = UpperConfidenceBoundGenerator(vocs=TEST_VOCS_BASE.model_dump())
        gen2.use_cuda = use_cuda
        check_generator_tensor_locations(gen2, device_map[use_cuda])

        with pytest.raises(ValueError):
            UpperConfidenceBoundGenerator(
                vocs=TEST_VOCS_BASE.model_dump(),
                log_transform_acquisition_function=True,
            )

    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_generate_c(self, use_cuda):
        gen = UpperConfidenceBoundGenerator(
            vocs=TEST_VOCS_BASE,
        )
        set_options(gen, use_cuda, add_data=True)

        candidate = generate_without_warnings(gen, 1)
        assert len(candidate) == 1

        # Will fail to converge most of the time - log softplus for q=2 is really unstable
        with warnings.catch_warnings(record=True) as w:
            candidate = gen.generate(2)
            assert sum(issubclass(x.category, GeneratorWarning) for x in w) == 1
        assert len(candidate) == 2

        assert isinstance(gen.computation_time, pd.DataFrame)
        assert len(gen.computation_time) == 2

        check_generator_tensor_locations(gen, device_map[use_cuda])

    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_generate_nc(self, use_cuda):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.constraints = {}
        gen = UpperConfidenceBoundGenerator(
            vocs=vocs,
        )
        set_options(gen, use_cuda, add_data=True)

        candidate = generate_without_warnings(gen, 1)
        assert len(candidate) == 1

        candidate = generate_without_warnings(gen, 2)
        assert len(candidate) == 2

        assert isinstance(gen.computation_time, pd.DataFrame)
        assert len(gen.computation_time) == 2

        check_generator_tensor_locations(gen, device_map[use_cuda])

    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_get_optimum(self, use_cuda):
        # Putting optimum test here for now since need an actual optimizer to fully test model fit
        vocs = TEST_VOCS_BASE
        gen = UpperConfidenceBoundGenerator(
            vocs=vocs,
        )
        set_options(gen, use_cuda=False, add_data=True)
        evaluator = Evaluator(function=xtest_callable)
        X = Xopt(generator=gen, evaluator=evaluator, vocs=vocs)
        X.random_evaluate(10)
        for _ in range(1):
            X.step()
        X.generator.get_optimum()

    def test_generate_w_overlapping_objectives_constraints(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.constraints = {"y1": ["GREATER_THAN", 0.0]}
        gen = UpperConfidenceBoundGenerator(
            vocs=test_vocs,
        )
        set_options(gen, add_data=True)

        candidate = gen.generate(1)
        assert len(candidate) == 1

    def test_in_xopt(self):
        evaluator = Evaluator(function=xtest_callable)
        gen = UpperConfidenceBoundGenerator(
            vocs=TEST_VOCS_BASE,
        )
        set_options(gen)

        X = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)
        X.random_evaluate(5)
        for _ in range(2):
            X.step()

    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_fixed_feature(self, use_cuda):
        # test with fixed feature not in vocs
        gen = UpperConfidenceBoundGenerator(vocs=TEST_VOCS_BASE)
        gen.fixed_features = {"p": 3.0}
        set_options(gen, use_cuda)
        data = deepcopy(TEST_VOCS_DATA)
        data["p"] = np.random.rand(len(data))

        gen.add_data(data)
        candidate = gen.generate(1)[0]
        assert candidate["p"] == 3

        check_generator_tensor_locations(gen, device_map[use_cuda])
        gen.get_optimum()

        # test with fixed feature in vocs
        data = deepcopy(TEST_VOCS_DATA)
        gen = UpperConfidenceBoundGenerator(vocs=TEST_VOCS_BASE)
        gen.fixed_features = {"x1": 3.0}
        set_options(gen, use_cuda)

        gen.add_data(data)
        candidate = gen.generate(1)[0]
        assert candidate["x1"] == 3

        check_generator_tensor_locations(gen, device_map[use_cuda])
        gen.get_optimum()

    def test_constraints_warning(self):
        with pytest.warns(UserWarning):
            _ = UpperConfidenceBoundGenerator(
                vocs=TEST_VOCS_BASE,
            )

    def test_negative_acq_values_warning(self):
        X = Xopt.from_yaml(
            """
            generator:
              name: upper_confidence_bound

            evaluator:
              function: xopt.resources.test_functions.sinusoid_1d.evaluate_sinusoid

            vocs:
              variables:
                x1: [0, 6.28]
              constraints:
                c1: [LESS_THAN, 0.0]
              objectives:
                y1: 'MAXIMIZE'
            """
        )
        _ = X.random_evaluate(10, seed=0)
        test_x = torch.linspace(*X.vocs.variables["x1"], 10, dtype=torch.float64)
        model = X.generator.train_model(X.data)
        acq = X.generator.get_acquisition(model)
        with pytest.warns(UserWarning):
            _ = acq(test_x.unsqueeze(1).unsqueeze(1))
