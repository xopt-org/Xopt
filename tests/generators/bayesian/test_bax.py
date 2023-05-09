from copy import deepcopy

import pandas as pd
import pytest
import torch

from xopt.base import Xopt

from xopt.evaluator import Evaluator
from xopt.generators.bayesian.bax_generator import BaxGenerator, BaxOptions
from xopt.generators.bayesian.options import BayesianOptions

from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA, xtest_callable


class TestBaxGenerator:
    def test_init(self):
        bax_gen = BaxGenerator(
            TEST_VOCS_BASE,
        )
        bax_gen.options.dict()
        # bax_gen.options.schema()

        with pytest.raises(ValueError):
            BaxGenerator(TEST_VOCS_BASE, BayesianOptions())

    def test_generate(self):
        gen = BaxGenerator(
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
        gen = BaxGenerator(
            TEST_VOCS_BASE,
        )

        if torch.cuda.is_available():
            gen.options.use_cuda = True
            gen.options.optim.raw_samples = 1
            gen.options.optim.num_restarts = 1
            gen.options.acq.monte_carlo_samples = 1
            gen.data = TEST_VOCS_DATA
            gen.construct_algo()

            candidate = gen.generate(1)
            assert len(candidate) == 1

    def test_generate_w_overlapping_objectives_constraints(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.constraints = {"y1": ["GREATER_THAN", 0.0]}
        gen = BaxGenerator(
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
        bax_gen = BaxGenerator(
            TEST_VOCS_BASE,
        )
        bax_gen.options.optim.raw_samples = 1
        bax_gen.options.optim.num_restarts = 1

        xopt = Xopt(generator=bax_gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)

        # initialize with single initial candidate
        xopt.step()

        # now use bayes opt
        for _ in range(1):
            xopt.step()

    def test_in_xopt_w_proximal(self):
        evaluator = Evaluator(function=xtest_callable)
        bax_gen = BaxGenerator(
            TEST_VOCS_BASE,
        )
        bax_gen.options.optim.raw_samples = 1
        bax_gen.options.optim.num_restarts = 1
        bax_gen.options.acq.proximal_lengthscales = [1.0, 1.0]

        xopt = Xopt(generator=bax_gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)

        # initialize with single initial candidate
        xopt.step()

        # now use bayes opt
        for _ in range(1):
            xopt.step()

    def test_positivity(self):
        # for BAX to work properly with constraints, it must always be positive.
        # to acheive this we set infeasible cost
        bax_gen = BaxGenerator(
            TEST_VOCS_BASE,
        )
        bax_gen.add_data(
            pd.DataFrame({"x1": -1.0, "x2": -1.0, "y1": 100.0, "c1": -100}, index=[0])
        )
        bax_gen.train_model()
        # evaluate acqf
        acqf = bax_gen.get_acquisition(bax_gen.model)
        with torch.no_grad():
            assert acqf(torch.tensor((-1.0, -1.0)).reshape(1, 1, 2)) >= 0.0
