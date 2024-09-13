from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import PeriodicKernel

from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.resources.test_functions.sinusoid_1d import evaluate_sinusoid, sinusoid_vocs
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA


class TestBayesianGenerator(TestCase):
    @patch.multiple(BayesianGenerator, __abstractmethods__=set())
    def test_init(self):
        gen = BayesianGenerator(vocs=TEST_VOCS_BASE)
        gen.model_dump()

    @patch.multiple(BayesianGenerator, __abstractmethods__=set())
    def test_get_model(self):
        test_data = deepcopy(TEST_VOCS_DATA)
        gen = BayesianGenerator(vocs=TEST_VOCS_BASE)

        model = gen.train_model(test_data)
        assert isinstance(model, GPyTorchModel)

        # test evaluating the model
        test_pts = torch.tensor(
            pd.DataFrame(
                TEST_VOCS_BASE.random_inputs(5, include_constants=False)
            ).to_numpy()
        )

        with torch.no_grad():
            model.posterior(test_pts)

        # test with prior model
        constructor = deepcopy(gen.gp_constructor)
        constructor.covar_modules = {"y1": PeriodicKernel()}

        gen = BayesianGenerator(vocs=TEST_VOCS_BASE, gp_constructor=constructor)
        model = gen.train_model(test_data)
        assert isinstance(model.models[0].covar_module, PeriodicKernel)

        # test with dict arguments
        gen = deepcopy(gen)
        gen.gp_constructor.covar_modules = {"y1": PeriodicKernel()}

        gen = BayesianGenerator(vocs=TEST_VOCS_BASE, **gen.model_dump())
        model = gen.train_model(test_data)
        assert isinstance(model.models[0].covar_module, PeriodicKernel)

    @patch.multiple(BayesianGenerator, __abstractmethods__=set())
    def test_get_model_w_conditions(self):
        # test with maximize and minimize
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives["y1"] = "MINIMIZE"
        test_data = deepcopy(TEST_VOCS_DATA)
        gen = BayesianGenerator(vocs=test_vocs)
        model = gen.train_model(test_data)
        assert torch.allclose(
            model.models[0].outcome_transform(torch.tensor(test_data["y1"].to_numpy()))[
                0
            ],
            model.train_targets[0],
        )

        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives["y1"] = "MAXIMIZE"
        test_data = deepcopy(TEST_VOCS_DATA)
        gen = BayesianGenerator(vocs=test_vocs)
        model = gen.train_model(test_data)
        assert torch.allclose(
            model.models[0].outcome_transform(torch.tensor(test_data["y1"].to_numpy()))[
                0
            ],
            model.train_targets[0],
        )

        # try with input data that contains Nans due to xopt raising an error
        # currently we drop all rows containing Nans
        test_data = deepcopy(TEST_VOCS_DATA)
        test_data["y1"].iloc[5] = np.nan
        model = gen.train_model(test_data)
        assert len(model.models[0].train_inputs[0]) == len(test_data) - 1

        # test with input data that is only Nans
        test_data = deepcopy(TEST_VOCS_DATA)
        test_data["y1"].iloc[:] = np.nan
        with pytest.raises(ValueError):
            gen.train_model(test_data)

        # test with the same objective and constraint keys
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.constraints = {"y1": ["GREATER_THAN", 0]}
        gen2 = BayesianGenerator(vocs=test_vocs)
        test_data = deepcopy(TEST_VOCS_DATA)
        gen2.train_model(test_data)

        # test with GPU if available
        if torch.cuda.is_available():
            test_vocs = deepcopy(TEST_VOCS_BASE)
            test_vocs.constraints = {"y1": ["GREATER_THAN", 0]}
            gen3 = BayesianGenerator(vocs=test_vocs)
            gen3.use_cuda = True
            test_data = deepcopy(TEST_VOCS_DATA)
            model = gen3.train_model(test_data)
            assert model.models[0].train_targets.is_cuda

    @patch.multiple(BayesianGenerator, __abstractmethods__=set())
    def test_transforms(self):
        gen = BayesianGenerator(vocs=sinusoid_vocs)
        evaluator = Evaluator(function=evaluate_sinusoid)
        X = Xopt(generator=gen, evaluator=evaluator, vocs=sinusoid_vocs)

        # generate some data samples
        import numpy as np

        test_samples = pd.DataFrame(np.linspace(0, 3.14, 10), columns=["x1"])
        X.evaluate_data(test_samples)

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

        # constraint transform - bilog
        # C = torch.from_numpy(X.data["c1"].to_numpy())
        # assert torch.allclose(
        #    model.train_targets[1], torch.sign(-C) * torch.log(1 + torch.abs(-C))
        # )

    # TODO: test passing UCB options to bayesian exploration

    @patch.multiple(BayesianGenerator, __abstractmethods__=set())
    def test_get_bounds(self):
        gen = BayesianGenerator(vocs=TEST_VOCS_BASE)
        bounds = gen._get_optimization_bounds()
        assert torch.allclose(bounds, torch.tensor(TEST_VOCS_BASE.bounds))

        # test with max_travel_distances specified but no data
        gen = BayesianGenerator(vocs=TEST_VOCS_BASE)
        gen.max_travel_distances = [0.1, 0.2]
        with pytest.raises(ValueError):
            gen._get_optimization_bounds()

        # test with max_travel_distances specified and data
        gen = BayesianGenerator(vocs=TEST_VOCS_BASE)
        gen.max_travel_distances = [0.1, 0.2]
        gen.add_data(pd.DataFrame({"x1": [0.5], "x2": [5.0], "y1": [0.5], "c1": [0.5]}))
        bounds = gen._get_optimization_bounds()
        assert torch.allclose(bounds, torch.tensor([[0.4, 3.0], [0.6, 7.0]]).to(bounds))

        # test with max_travel_distances specified and data
        high_d_vocs = deepcopy(TEST_VOCS_BASE)
        high_d_vocs.variables["x3"] = [0, 1]

        gen = BayesianGenerator(vocs=high_d_vocs)
        gen.max_travel_distances = [0.1, 0.2, 0.1]
        gen.add_data(
            pd.DataFrame(
                {"x1": [0.5], "x2": [5.0], "x3": [0.5], "y1": [0.5], "c1": [0.5]}
            )
        )
        bounds = gen._get_optimization_bounds()
        assert torch.allclose(
            bounds, torch.tensor([[0.4, 3.0, 0.4], [0.6, 7.0, 0.6]]).to(bounds)
        )

        # test with bad max_distances
        gen = BayesianGenerator(vocs=high_d_vocs)
        gen.max_travel_distances = [0.1, 0.2]
        gen.add_data(
            pd.DataFrame(
                {"x1": [0.5], "x2": [5.0], "x3": [0.5], "y1": [0.5], "c1": [0.5]}
            )
        )
        with pytest.raises(ValueError):
            gen._get_optimization_bounds()

    @patch.multiple(BayesianGenerator, __abstractmethods__=set())
    def test_fixed_feature(self):
        # test with a fixed feature not contained in vocs
        gen = BayesianGenerator(vocs=TEST_VOCS_BASE)
        gen.fixed_features = {"p": 3.0}

        assert gen.model_input_names == [*TEST_VOCS_BASE.variable_names, "p"]

        data = deepcopy(TEST_VOCS_DATA)
        data["p"] = 5.0

        gen.add_data(data)
        model = gen.train_model()
        assert torch.allclose(
            model.models[0].input_transform.bounds,
            torch.tensor(((0, 0, 5), (1, 10, 5))).double(),
        )

        # test fixed_feature in vocs
        gen = BayesianGenerator(vocs=TEST_VOCS_BASE)
        gen.fixed_features = {"x1": 3.0}

        # test naming
        assert gen.model_input_names == TEST_VOCS_BASE.variable_names
        assert gen._candidate_names == ["x2"]

        # test get bounds
        bounds = gen._get_optimization_bounds()
        assert torch.allclose(bounds, torch.tensor((0.0, 10.0)).reshape(2, 1).double())

        data = deepcopy(TEST_VOCS_DATA)
        gen.add_data(data)
        model = gen.train_model()
        assert torch.allclose(
            model.models[0].input_transform.bounds,
            torch.tensor(((0, 0), (1, 10))).double(),
        )

        # test bad fixed feature name
        gen = BayesianGenerator(vocs=TEST_VOCS_BASE)
        gen.fixed_features = {"bad_name": 3.0}
        data = deepcopy(TEST_VOCS_DATA)
        gen.add_data(data)

        with pytest.raises(KeyError):
            gen.train_model()
