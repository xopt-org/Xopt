import pytest
import torch
from unittest.mock import patch
from unittest import TestCase

from botorch.models import ModelListGP

from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA


class TestBayesianGenerator(TestCase):
    @patch.multiple(BayesianGenerator, __abstractmethods__=set())
    def test_init(self):
        gen = BayesianGenerator(TEST_VOCS_BASE)

    @patch.multiple(BayesianGenerator, __abstractmethods__=set())
    def test_get_model(self):
        gen = BayesianGenerator(TEST_VOCS_BASE)
        model = gen.train_model(TEST_VOCS_DATA)
        assert isinstance(model, ModelListGP)

    @patch.multiple(BayesianGenerator, __abstractmethods__=set())
    def test_get_training_data(self):
        gen = BayesianGenerator(TEST_VOCS_BASE)
        inputs, outputs = gen.get_training_data(TEST_VOCS_DATA)
        inames = list(TEST_VOCS_BASE.variables.keys())
        onames = list(TEST_VOCS_BASE.objectives.keys()) + list(
            TEST_VOCS_BASE.constraints.keys()
        )

        true_inputs = torch.from_numpy(TEST_VOCS_DATA[inames].to_numpy())
        true_outputs = torch.from_numpy(TEST_VOCS_DATA[onames].to_numpy())
        assert torch.allclose(inputs, true_inputs)
        assert torch.allclose(outputs, true_outputs)
