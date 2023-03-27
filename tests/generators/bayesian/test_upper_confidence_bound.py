from copy import deepcopy

import pandas as pd
import pytest
import torch

from xopt.base import Xopt

from xopt.evaluator import Evaluator
from xopt.generators.bayesian.options import BayesianOptions
from xopt.generators.bayesian.upper_confidence_bound import (
    UpperConfidenceBoundGenerator,
)

from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA, xtest_callable


class TestUpperConfidenceBoundGenerator:
    def test_init(self):
        ucb_gen = UpperConfidenceBoundGenerator(TEST_VOCS_BASE)
        ucb_gen.options.dict()
        # ucb_gen.options.schema()

        with pytest.raises(ValueError):
            UpperConfidenceBoundGenerator(TEST_VOCS_BASE, BayesianOptions())

    def test_generate(self):
        gen = UpperConfidenceBoundGenerator(
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

    def test_cuda(self):
        gen = UpperConfidenceBoundGenerator(
            TEST_VOCS_BASE,
        )

        if torch.cuda.is_available():
            gen.options.use_cuda = True
            gen.options.optim.raw_samples = 1
            gen.options.optim.num_restarts = 1
            gen.options.acq.monte_carlo_samples = 1
            gen.data = TEST_VOCS_DATA

            candidate = gen.generate(1)
            assert len(candidate) == 1

    def test_generate_w_overlapping_objectives_constraints(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.constraints = {"y1": ["GREATER_THAN", 0.0]}
        gen = UpperConfidenceBoundGenerator(
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
        ucb_gen = UpperConfidenceBoundGenerator(
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
        ucb_gen = UpperConfidenceBoundGenerator(
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

    def test_positivity(self):
        # for UCB to work properly with constraints, it must always be positive.
        # to acheive this we set infeasible cost
        ucb_gen = UpperConfidenceBoundGenerator(
            TEST_VOCS_BASE,
        )
        ucb_gen.add_data(
            pd.DataFrame({"x1": -1.0, "x2": -1.0, "y1": 100.0, "c1": -100}, index=[0])
        )
        ucb_gen.train_model()
        # evaluate acqf
        acqf = ucb_gen.get_acquisition(ucb_gen.model)
        with torch.no_grad():
            assert acqf(torch.tensor((-1.0, -1.0)).reshape(1, 1, 2)) >= 0.0
