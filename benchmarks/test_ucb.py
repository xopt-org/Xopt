import os
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.upper_confidence_bound import (
    UpperConfidenceBoundGenerator,
)
from xopt.resources.testing import (
    TEST_VOCS_BASE,
    xtest_callable,
)

cuda_combinations = [False] if not torch.cuda.is_available() else [False, True]
device_map = {False: torch.device("cpu"), True: torch.device("cuda:0")}


def test_init_data_n(n):
    return {
        "x1": np.linspace(0.01, 1.0, n),
        "x2": np.linspace(0.01, 1.0, n) * 10.0,
        "constant1": 1.0,
    }


def set_options(gen, use_cuda=False, add_data=False, n_data=100):
    gen.use_cuda = use_cuda
    gen.numerical_optimizer.n_restarts = 2
    gen.n_monte_carlo_samples = 32
    x_data = test_init_data_n(n_data)
    data = pd.DataFrame({**x_data, **xtest_callable(x_data)})
    if add_data:
        gen.add_data(data)


class TestUpperConfidenceBoundGenerator:
    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_ucb_c(self, use_cuda):
        gen = UpperConfidenceBoundGenerator(
            vocs=TEST_VOCS_BASE,
        )
        set_options(gen, use_cuda, add_data=True)
        for _ in range(200):
            gen.generate(1)

    def test_ucb_nc(self, use_cuda):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.constraints = {}
        gen = UpperConfidenceBoundGenerator(
            vocs=vocs,
        )
        set_options(gen, use_cuda, add_data=True)
        for _ in range(200):
            gen.generate(1)

    def test_ucb_c_xopt(self):
        os.environ["OMP_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        evaluator = Evaluator(function=xtest_callable)
        gen = UpperConfidenceBoundGenerator(
            vocs=TEST_VOCS_BASE,
        )
        set_options(gen)

        X = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)
        X.random_evaluate(100)
        for _ in range(200):
            X.step()
