import torch
import numpy as np
from copy import deepcopy
import pytest
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement

from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.expected_improvement import ExpectedImprovementGenerator
from xopt.generators.bayesian.upper_confidence_bound import UCBOptions
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA, xtest_callable
from xopt.resources.test_functions.sinusoid_1d import sinusoid_vocs, evaluate_sinusoid


class TestExpectedImprovement:
    def test_init(self):
        ei_gen = ExpectedImprovementGenerator(TEST_VOCS_BASE)
        ei_gen.options.dict()

        with pytest.raises(ValueError):
            ExpectedImprovementGenerator(TEST_VOCS_BASE, UCBOptions())

    def test_generate(self):
        gen = ExpectedImprovementGenerator(
            TEST_VOCS_BASE,
        )
        gen.options.optim.raw_samples = 1
        gen.options.optim.num_restarts = 1
        gen.options.acq.monte_carlo_samples = 1
        gen.data = TEST_VOCS_DATA

        candidate = gen.generate(1)
        assert len(candidate) == 1

        # candidate = gen.generate(2)
        # assert len(candidate) == 2

    def test_generate_w_overlapping_objectives_constraints(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.constraints = {"y1": ["GREATER_THAN", 0.0]}
        gen = ExpectedImprovementGenerator(
            test_vocs,
        )
        gen.options.optim.raw_samples = 1
        gen.options.optim.num_restarts = 1
        gen.options.acq.monte_carlo_samples = 1
        gen.data = TEST_VOCS_DATA

        candidate = gen.generate(1)
        assert len(candidate) == 1

    def test_in_xopt(self):
        evaluator = Evaluator(function=xtest_callable)
        ucb_gen = ExpectedImprovementGenerator(
            TEST_VOCS_BASE,
        )
        ucb_gen.options.optim.raw_samples = 1
        ucb_gen.options.optim.num_restarts = 1

        xopt = Xopt(generator=ucb_gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)

        # initialize with single initial candidate
        xopt.step()

        # now use bayes opt
        for _ in range(1):
            xopt.step()

    def test_in_xopt_w_proximal(self):
        evaluator = Evaluator(function=xtest_callable)
        ucb_gen = ExpectedImprovementGenerator(
            TEST_VOCS_BASE,
        )
        ucb_gen.options.optim.raw_samples = 1
        ucb_gen.options.optim.num_restarts = 1
        ucb_gen.options.acq.proximal_lengthscales = [1.0, 1.0]

        xopt = Xopt(generator=ucb_gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)

        # initialize with single initial candidate
        xopt.step()

        # now use bayes opt
        for _ in range(1):
            xopt.step()

    def test_acquisition_accuracy(self):
        # fix random seeds for BoTorch
        torch.manual_seed(0)
        np.random.seed(0)

        vocs = sinusoid_vocs
        vocs.constraints = {}
        for objective in ["MINIMIZE", "MAXIMIZE"]:
            vocs.objectives["y1"] = objective
            evaluator = Evaluator(function=evaluate_sinusoid)
            generator = ExpectedImprovementGenerator(vocs)
            X = Xopt(evaluator=evaluator, generator=generator, vocs=vocs)
            X.step()

            distance = 0.0
            for i in range(3):
                model = X.generator.train_model()

                # analytical acquisition
                if objective == "MAXIMIZE":
                    maximize = True
                    best_f = torch.tensor(X.data["y1"].values).max()
                else:
                    maximize = False
                    best_f = torch.tensor(X.data["y1"].values).min()
                acq_analytical = ExpectedImprovement(model, best_f=best_f,
                                                     maximize=maximize)
                candidate_analytical, _ = optimize_acqf(
                    acq_function=acq_analytical,
                    bounds=torch.tensor(vocs.bounds),
                    q=1,
                    num_restarts=20,
                    raw_samples=100
                )

                # xopt step
                X.step()

                # calculate distance from analytical candidate
                distance += torch.abs(
                    X.data["x1"].values[-1] - candidate_analytical.squeeze())

            # distance should be small
            assert distance < 0.1
