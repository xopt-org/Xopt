from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

import pytest
import torch

from xopt.generators.bayesian.turbo import TuRBOBayesianGenerator
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA


class TestBayesianGenerator(TestCase):
    @patch.multiple(TuRBOBayesianGenerator, __abstractmethods__=set())
    def test_init(self):
        TuRBOBayesianGenerator(TEST_VOCS_BASE)

    @patch.multiple(TuRBOBayesianGenerator, __abstractmethods__=set())
    def test_get_trust_region(self):
        # test in 1D
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.variables = {"x1": [0, 1]}
        gen = TuRBOBayesianGenerator(test_vocs)
        gen.add_data(TEST_VOCS_DATA)

        with pytest.raises(RuntimeError):
            gen.get_trust_region()

        gen.train_model()
        bounds = gen._get_bounds()
        tr = gen.get_trust_region()
        assert torch.all(tr[0] >= bounds[0])
        assert torch.all(tr[1] <= bounds[1])

        # test in 2D
        gen = TuRBOBayesianGenerator(TEST_VOCS_BASE)
        gen.add_data(TEST_VOCS_DATA)

        with pytest.raises(RuntimeError):
            gen.get_trust_region()

        gen.train_model()
        bounds = gen._get_bounds()
        tr = gen.get_trust_region()
        assert torch.all(tr[0] >= bounds[0])
        assert torch.all(tr[1] <= bounds[1])
