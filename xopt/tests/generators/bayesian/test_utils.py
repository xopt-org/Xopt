import time
from copy import deepcopy

import pytest
import torch

from xopt import Evaluator, Xopt
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
from xopt.generators.bayesian.objectives import create_mobo_objective
from xopt.generators.bayesian.utils import jit_gp_model
from xopt.resources.testing import TEST_VOCS_BASE, xtest_callable

cuda_combinations = [False] if not torch.cuda.is_available() else [False, True]
device_map = {False: torch.device("cpu"), True: torch.device("cuda:0")}


class TestUtils:
    def test_mobo_objective(self):
        test_vocs_copy = deepcopy(TEST_VOCS_BASE)
        test_vocs_copy.objectives["y2"] = "MAXIMIZE"
        obj = create_mobo_objective(test_vocs_copy)

        # test large sample shape
        test_samples = torch.randn(3, 4, 5, 3).double()
        output = obj(test_samples)
        assert output.shape == torch.Size([3, 4, 5, 2])

        # test to make sure values are correct - minimize axis should be negated
        test_samples = torch.rand(5, 4, 3)
        output = obj(test_samples)
        assert torch.allclose(output[..., 1], test_samples[..., 1])
        assert torch.allclose(output[..., 0], -test_samples[..., 0])

    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_jit(self, use_cuda):
        evaluator = Evaluator(function=xtest_callable)
        gen = UpperConfidenceBoundGenerator(
            vocs=TEST_VOCS_BASE,
        )
        gen.use_cuda = use_cuda
        gen.numerical_optimizer.n_restarts = 2
        gen.n_monte_carlo_samples = 4
        X = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)
        X.random_evaluate(1000)
        for _ in range(1):
            X.step()
        gen = X.generator

        t1 = time.perf_counter()
        model_jit = jit_gp_model(gen.model.models[0], gen.vocs, gen.tkwargs).to(
            device_map[use_cuda]
        )
        t2 = time.perf_counter()
        print(f"JIT compile: {t2 - t1:.4f} seconds")

        x_grid = torch.tensor(
            gen.vocs.grid_inputs(50, include_constants=False).to_numpy()
        )
        x_grid = x_grid.to(device_map[use_cuda])

        t1 = time.perf_counter()
        gen.model.models[0](x_grid)
        t2 = time.perf_counter()
        print(f"Original time: {t2 - t1:.4f} seconds")

        t1 = time.perf_counter()
        model_jit(x_grid)
        t2 = time.perf_counter()
        print(f"JIT time: {t2 - t1:.4f} seconds")
