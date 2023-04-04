from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

import torch

from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.turbo import get_trust_region, TurboState
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA


class TestTurbo(TestCase):
    @patch.multiple(BayesianGenerator, __abstractmethods__=set())
    def test_get_trust_region(self):
        # test in 1D
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.variables = {"x1": [0, 1]}
        gen = BayesianGenerator(test_vocs)
        gen.add_data(TEST_VOCS_DATA)
        model = gen.train_model()
        bounds = gen._get_bounds()

        turbo_state = TurboState(gen.vocs.n_variables, 1)

        tr = get_trust_region(gen.vocs, model, bounds, gen.data, turbo_state, {})
        assert torch.all(tr[0] >= bounds[0])
        assert torch.all(tr[1] <= bounds[1])

        # test in 2D
        test_vocs = deepcopy(TEST_VOCS_BASE)
        gen = BayesianGenerator(test_vocs)
        gen.add_data(TEST_VOCS_DATA)
        model = gen.train_model()
        bounds = gen._get_bounds()

        turbo_state = TurboState(gen.vocs.n_variables, 1)

        tr = get_trust_region(gen.vocs, model, bounds, gen.data, turbo_state, {})

        assert torch.all(tr[0] >= bounds[0])
        assert torch.all(tr[1] <= bounds[1])
