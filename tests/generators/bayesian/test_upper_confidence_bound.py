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
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA, xtest_callable


class TestUpperConfidenceBoundGenerator:
    def test_init(self):
        ucb_gen = UpperConfidenceBoundGenerator(vocs=TEST_VOCS_BASE)
        ucb_gen.model_dump_json()

        # test init from dict
        UpperConfidenceBoundGenerator(vocs=TEST_VOCS_BASE.dict())

        with pytest.raises(ValueError):
            UpperConfidenceBoundGenerator(
                vocs=TEST_VOCS_BASE.dict(), log_transform_acquisition_function=True
            )

    def test_generate(self):
        gen = UpperConfidenceBoundGenerator(
            vocs=TEST_VOCS_BASE,
        )
        gen.numerical_optimizer.n_restarts = 1
        gen.n_monte_carlo_samples = 1
        gen.data = TEST_VOCS_DATA

        candidate = gen.generate(1)
        assert len(candidate) == 1

        candidate = gen.generate(2)
        assert len(candidate) == 2

        # test time tracking
        assert isinstance(gen.computation_time, pd.DataFrame)
        assert len(gen.computation_time) == 2

    def test_cuda(self):
        gen = UpperConfidenceBoundGenerator(
            vocs=TEST_VOCS_BASE,
        )

        if torch.cuda.is_available():
            gen.use_cuda = True
            gen.numerical_optimizer.n_restarts = 1
            gen.n_monte_carlo_samples = 1
            gen.data = TEST_VOCS_DATA

            candidate = gen.generate(1)
            assert len(candidate) == 1

    def test_generate_w_overlapping_objectives_constraints(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.constraints = {"y1": ["GREATER_THAN", 0.0]}
        gen = UpperConfidenceBoundGenerator(
            vocs=test_vocs,
        )
        gen.numerical_optimizer.n_restarts = 1
        gen.n_monte_carlo_samples = 1
        gen.data = TEST_VOCS_DATA

        candidate = gen.generate(1)
        assert len(candidate) == 1

    def test_in_xopt(self):
        evaluator = Evaluator(function=xtest_callable)
        gen = UpperConfidenceBoundGenerator(
            vocs=TEST_VOCS_BASE,
        )
        gen.numerical_optimizer.n_restarts = 1
        gen.n_monte_carlo_samples = 1

        X = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)

        # initialize with single initial candidate
        X.random_evaluate(1)

        # now use bayes opt
        for _ in range(1):
            X.step()

    def test_positivity(self):
        # for UCB to work properly with constraints, it must always be positive.
        # to acheive this we set infeasible cost
        ucb_gen = UpperConfidenceBoundGenerator(
            vocs=TEST_VOCS_BASE,
        )
        ucb_gen.add_data(
            pd.DataFrame({"x1": -1.0, "x2": -1.0, "y1": 100.0, "c1": -100}, index=[0])
        )
        ucb_gen.train_model()
        # evaluate acqf
        acqf = ucb_gen.get_acquisition(ucb_gen.model)
        with torch.no_grad():
            assert acqf(torch.tensor((-1.0, -1.0)).reshape(1, 1, 2)) >= 0.0

    def test_fixed_feature(self):
        # test with fixed feature not in vocs
        gen = UpperConfidenceBoundGenerator(vocs=TEST_VOCS_BASE)
        gen.fixed_features = {"p": 3.0}
        gen.n_monte_carlo_samples = 1
        gen.numerical_optimizer.n_restarts = 1
        data = deepcopy(TEST_VOCS_DATA)
        data["p"] = np.random.rand(len(data))

        gen.add_data(data)
        candidate = gen.generate(1)[0]
        assert candidate["p"] == 3

        # test with fixed feature in vocs
        gen = UpperConfidenceBoundGenerator(vocs=TEST_VOCS_BASE)
        gen.fixed_features = {"x1": 3.0}
        gen.n_monte_carlo_samples = 1
        gen.numerical_optimizer.n_restarts = 1

        gen.add_data(data)
        candidate = gen.generate(1)[0]
        assert candidate["x1"] == 3

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
        test_x = torch.linspace(*X.vocs.variables["x1"], 10)
        model = X.generator.train_model(X.data)
        acq = X.generator.get_acquisition(model)
        with pytest.warns(UserWarning):
            _ = acq(test_x.unsqueeze(1).unsqueeze(1))
