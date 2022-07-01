from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms import Normalize, Standardize

from xopt.evaluator import Evaluator
from xopt.base import Xopt
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.resources.test_functions.sinusoid_1d import evaluate_sinusoid, sinusoid_vocs
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA


class TestBayesianGenerator(TestCase):
    @patch.multiple(BayesianGenerator, __abstractmethods__=set())
    def test_init(self):
        BayesianGenerator(TEST_VOCS_BASE)

    @patch.multiple(BayesianGenerator, __abstractmethods__=set())
    def test_get_model(self):
        test_data = deepcopy(TEST_VOCS_DATA)
        gen = BayesianGenerator(TEST_VOCS_BASE)
        model = gen.train_model(test_data)
        assert isinstance(model, GPyTorchModel)

        # test evaluating the model
        test_pts = torch.tensor(
            pd.DataFrame(TEST_VOCS_BASE.random_inputs(5, False, False)).to_numpy()
        )
        with torch.no_grad():
            model(test_pts)

        # try with input data that contains Nans due to xopt raising an error
        # currently we drop all rows containing Nans
        test_data["y1"].iloc[5] = np.NaN
        model = gen.train_model(test_data)
        assert len(model.models[0].train_inputs[0]) == len(test_data) - 1

    @patch.multiple(BayesianGenerator, __abstractmethods__=set())
    def test_transforms(self):
        gen = BayesianGenerator(sinusoid_vocs)
        evaluator = Evaluator(function=evaluate_sinusoid)
        X = Xopt(generator=gen, evaluator=evaluator, vocs=sinusoid_vocs)

        # generate some data samples
        import numpy as np

        test_samples = pd.DataFrame(np.linspace(0, 2 * 3.14, 10), columns=["x1"])
        X.submit_data(test_samples)

        # create gp model with data
        model = gen.train_model(X.data)

        # test input normalization
        input_transform = Normalize(1, bounds=torch.tensor(sinusoid_vocs.bounds))
        for inputs in model.train_inputs:
            assert torch.allclose(
                inputs[0], input_transform(torch.from_numpy(X.data["x1"].to_numpy())).T
            )

        # test outcome transform(s)
        # objective transform - standardization
        outcome_transform = Standardize(1)
        assert torch.allclose(
            model.train_targets[0],
            torch.flatten(
                outcome_transform(
                    torch.from_numpy(X.data["y1"].to_numpy()).unsqueeze(-1)
                )[0]
            ),
        )

        # constraint transform
        C = torch.from_numpy(X.data["c1"].to_numpy())
        C = C / torch.sqrt(torch.sum(C**2) / C.numel())  # standardization
        assert torch.allclose(
            model.train_targets[1], torch.sign(-C) * torch.log(1 + torch.abs(-C))
        )

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
