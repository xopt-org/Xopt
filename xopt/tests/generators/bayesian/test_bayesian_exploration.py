from copy import deepcopy

import pytest
import torch

from xopt.base import Xopt
from xopt.errors import VOCSError
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.bayesian_exploration import BayesianExplorationGenerator
from xopt.resources.testing import (
    TEST_VOCS_BASE,
    TEST_VOCS_DATA,
    create_set_options_helper,
    xtest_callable,
)

cuda_combinations = [False] if not torch.cuda.is_available() else [False, True]
device_map = {False: torch.device("cpu"), True: torch.device("cuda:0")}


set_options = create_set_options_helper(data=TEST_VOCS_DATA)


class TestBayesianExplorationGenerator:
    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_generate(self, use_cuda):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs2 = deepcopy(test_vocs)
        test_vocs2.objectives = {}
        test_vocs2.observables = ["y1"]

        for ele in [test_vocs2]:
            gen = BayesianExplorationGenerator(
                vocs=ele,
            )
            set_options(gen, use_cuda, add_data=True)

            candidate = gen.generate(1)
            assert len(candidate) == 1
            candidate = gen.generate(5)
            assert len(candidate) == 5

            # test without constraints
            gen = BayesianExplorationGenerator(
                vocs=ele,
            )
            set_options(gen, use_cuda, add_data=True)

            candidate = gen.generate(1)
            assert len(candidate) == 1
            candidate = gen.generate(5)
            assert len(candidate) == 5

    def test_in_xopt(self):
        evaluator = Evaluator(function=xtest_callable)

        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives = {}
        test_vocs.observables = ["y1"]

        gen = BayesianExplorationGenerator(vocs=test_vocs)
        gen.numerical_optimizer.n_restarts = 1
        gen.n_monte_carlo_samples = 1
        gen.data = TEST_VOCS_DATA

        X = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)

        # now use bayes opt
        X.step()
        X.step()

    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_with_turbo(self, use_cuda):
        evaluator = Evaluator(function=xtest_callable)

        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives = {}
        test_vocs.observables = ["y1"]

        gen = BayesianExplorationGenerator(
            vocs=test_vocs, turbo_controller="SafetyTurboController"
        )
        set_options(gen, use_cuda, add_data=True)

        X = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)

        # now use bayes opt
        X.step()
        X.step()

    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_interpolation(self, use_cuda):
        evaluator = Evaluator(function=xtest_callable)

        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives = {}
        test_vocs.observables = ["y1"]

        gen = BayesianExplorationGenerator(vocs=test_vocs)
        set_options(gen, use_cuda)
        gen.n_interpolate_points = 5

        X = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)
        X.add_data(TEST_VOCS_DATA)

        # now use bayes opt
        X.step()
        X.step()
        assert len(X.data) == 20

    def test_vocs_validation(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)

        with pytest.raises(VOCSError):
            BayesianExplorationGenerator(vocs=test_vocs)
