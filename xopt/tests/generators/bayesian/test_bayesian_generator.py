from copy import deepcopy
import os
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import Module
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import PeriodicKernel

from gest_api.vocs import ContinuousVariable

from xopt import VOCS
from xopt.base import Xopt
from xopt.errors import VOCSError, XoptError
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.bayesian.base_model import ModelConstructor
from xopt.generators.bayesian.bayesian_generator import (
    BayesianGenerator,
    MultiObjectiveBayesianGenerator,
)
from xopt.numerical_optimizer import GridOptimizer, LBFGSOptimizer
from xopt.pydantic import encode_torch_module
from xopt.resources.test_functions.sinusoid_1d import evaluate_sinusoid, sinusoid_vocs
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA
from xopt.vocs import random_inputs


class PatchBayesianGenerator(BayesianGenerator):
    """
    Patch the Bayesian Generator class to allow for testing.
    """

    supports_batch_generation: bool = True
    # supports_multi_objective: bool = True
    supports_single_objective: bool = True
    supports_constraints: bool = True


class MultiObjectivePatchBayesianGenerator(MultiObjectiveBayesianGenerator):
    supports_constraints: bool = True


class DummyModelConstructor(ModelConstructor):
    name: str = "dummy"

    def build_model(self, *a, **k):
        pass  # pragma: no cover

    def build_model_from_vocs(self, *a, **k):
        pass  # pragma: no cover

    def build_single_task_gp(self, *a, **k):
        pass  # pragma: no cover


class TestBayesianGenerator(TestCase):
    @patch.multiple(PatchBayesianGenerator, __abstractmethods__=set())
    def test_init(self):
        gen = PatchBayesianGenerator(vocs=TEST_VOCS_BASE)
        gen.model_dump()

    @patch.multiple(PatchBayesianGenerator, __abstractmethods__=set())
    def test_get_model(self):
        test_data = deepcopy(TEST_VOCS_DATA)
        gen = PatchBayesianGenerator(vocs=TEST_VOCS_BASE)

        model = gen.train_model(test_data)
        assert isinstance(model, GPyTorchModel)

        # test evaluating the model
        test_pts = torch.tensor(
            pd.DataFrame(
                random_inputs(TEST_VOCS_BASE, 5, include_constants=False)
            ).to_numpy()
        )

        with torch.no_grad():
            model.posterior(test_pts)

        # test with prior model
        constructor = deepcopy(gen.gp_constructor)
        constructor.covar_modules = {"y1": PeriodicKernel()}

        gen = PatchBayesianGenerator(vocs=TEST_VOCS_BASE, gp_constructor=constructor)
        model = gen.train_model(test_data)
        assert isinstance(model.models[0].covar_module, PeriodicKernel)

        # test with dict arguments
        gen = deepcopy(gen)
        gen.gp_constructor.covar_modules = {"y1": PeriodicKernel()}

        gen = PatchBayesianGenerator(vocs=TEST_VOCS_BASE, **gen.model_dump())
        model = gen.train_model(test_data)
        assert isinstance(model.models[0].covar_module, PeriodicKernel)

    def test_class_methods(self):
        # test base class
        assert BayesianGenerator.get_compatible_turbo_controllers() == [None]
        assert BayesianGenerator.get_compatible_numerical_optimizers() == [
            LBFGSOptimizer,
            GridOptimizer,
        ]

        # define separate class
        class CustomBayesianGenerator(BayesianGenerator):
            _compatible_turbo_controllers = None

        assert CustomBayesianGenerator.get_compatible_turbo_controllers() == [None]

    def test_torch_module_validation(self):
        # test validate torch modules
        encoded_module = encode_torch_module(torch.nn.Linear(5, 2))
        exit_val = BayesianGenerator.validate_torch_modules("base64: " + encoded_module)
        assert isinstance(exit_val, Module)

        torch.save(torch.nn.Linear(3, 1), "test_module.pt")
        exit_val = BayesianGenerator.validate_torch_modules("test_module.pt")
        assert isinstance(exit_val, torch.nn.Linear)
        os.remove("test_module.pt")

        with pytest.raises(XoptError):
            BayesianGenerator.validate_torch_modules("invalid_string")

    def test_numerical_optimizer_validation(self):
        # test with None
        exit_val = BayesianGenerator.validate_numerical_optimizer(None)
        assert exit_val == LBFGSOptimizer()

        # test with class
        exit_val = BayesianGenerator.validate_numerical_optimizer(GridOptimizer())
        assert exit_val == GridOptimizer()

        # test with string
        exit_val = BayesianGenerator.validate_numerical_optimizer("LBFGS")
        assert exit_val == LBFGSOptimizer()

        with pytest.raises(ValueError):
            BayesianGenerator.validate_numerical_optimizer("NotAnOptimizer")

        # test with dict
        exit_val = BayesianGenerator.validate_numerical_optimizer({"name": "grid"})
        assert exit_val == GridOptimizer()

        with pytest.raises(ValueError):
            BayesianGenerator.validate_numerical_optimizer({"name": "NotAnOptimizer"})

        with pytest.raises(ValueError):
            BayesianGenerator.validate_numerical_optimizer(5)

    @patch.multiple(PatchBayesianGenerator, __abstractmethods__=set())
    def test_get_model_w_conditions(self):
        # test with maximize and minimize
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives["y1"] = "MINIMIZE"
        test_data = deepcopy(TEST_VOCS_DATA)
        gen = PatchBayesianGenerator(vocs=test_vocs)
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
        gen = PatchBayesianGenerator(vocs=test_vocs)
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
        test_data.loc[test_data.index[5], "y1"] = np.nan
        model = gen.train_model(test_data)
        assert len(model.models[0].train_inputs[0]) == len(test_data) - 1

        # test with input data that is only Nans
        test_data = deepcopy(TEST_VOCS_DATA)
        test_data.loc[:, "y1"] = np.nan
        with pytest.raises(ValueError):
            gen.train_model(test_data)

        # test with the same objective and constraint keys
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.constraints = {"y1": ["GREATER_THAN", 0]}
        gen2 = PatchBayesianGenerator(vocs=test_vocs)
        test_data = deepcopy(TEST_VOCS_DATA)
        gen2.train_model(test_data)

        # test with GPU if available
        if torch.cuda.is_available():
            test_vocs = deepcopy(TEST_VOCS_BASE)
            test_vocs.constraints = {"y1": ["GREATER_THAN", 0]}
            gen3 = PatchBayesianGenerator(vocs=test_vocs)
            gen3.use_cuda = True
            test_data = deepcopy(TEST_VOCS_DATA)
            model = gen3.train_model(test_data)
            assert model.models[0].train_targets.is_cuda

    @patch.multiple(PatchBayesianGenerator, __abstractmethods__=set())
    def test_transforms(self):
        gen = PatchBayesianGenerator(vocs=sinusoid_vocs)
        evaluator = Evaluator(function=evaluate_sinusoid)
        X = Xopt(generator=gen, evaluator=evaluator, vocs=sinusoid_vocs)

        # generate some data samples
        import numpy as np

        test_samples = pd.DataFrame(np.linspace(0, 3.14, 10), columns=["x1"])
        X.evaluate_data(test_samples)

        # create gp model with data
        model = gen.train_model(X.data)

        # test input normalization
        input_transform = Normalize(
            1, bounds=torch.tensor(sinusoid_vocs.bounds, dtype=torch.double).T
        )
        for inputs in model.train_inputs:
            assert torch.allclose(
                inputs[0].unsqueeze(-1).T,
                input_transform(
                    torch.from_numpy(X.data["x1"].to_numpy(copy=True)).unsqueeze(-1)
                ).T,
            )

        # test outcome transform(s)
        # objective transform - standardization
        outcome_transform = Standardize(1)
        assert torch.allclose(
            model.train_targets[0],
            torch.flatten(
                outcome_transform(
                    torch.from_numpy(X.data["y1"].to_numpy(copy=True)).unsqueeze(-1)
                )[0]
            ),
        )

        # constraint transform - bilog
        # C = torch.from_numpy(X.data["c1"].to_numpy())
        # assert torch.allclose(
        #    model.train_targets[1], torch.sign(-C) * torch.log(1 + torch.abs(-C))
        # )

    # TODO: test passing UCB options to bayesian exploration

    @patch.multiple(PatchBayesianGenerator, __abstractmethods__=set())
    def test_get_bounds(self):
        gen = PatchBayesianGenerator(vocs=TEST_VOCS_BASE)
        bounds = gen._get_optimization_bounds()
        assert torch.allclose(bounds, torch.tensor(TEST_VOCS_BASE.bounds).T.to(bounds))

        # test with max_travel_distances specified but no data
        gen = PatchBayesianGenerator(vocs=TEST_VOCS_BASE)
        gen.max_travel_distances = [0.1, 0.2]
        with pytest.raises(ValueError):
            gen._get_optimization_bounds()

        # test with max_travel_distances specified and data
        gen = PatchBayesianGenerator(vocs=TEST_VOCS_BASE)
        gen.max_travel_distances = [0.1, 0.2]
        gen.add_data(pd.DataFrame({"x1": [0.5], "x2": [5.0], "y1": [0.5], "c1": [0.5]}))
        bounds = gen._get_optimization_bounds()
        assert torch.allclose(bounds, torch.tensor([[0.4, 3.0], [0.6, 7.0]]).to(bounds))

        # test with max_travel_distances specified and data
        high_d_vocs = deepcopy(TEST_VOCS_BASE)
        high_d_vocs.variables["x3"] = ContinuousVariable(domain=[0, 1.0])

        gen = PatchBayesianGenerator(vocs=high_d_vocs)
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
        gen = PatchBayesianGenerator(vocs=high_d_vocs)
        gen.max_travel_distances = [0.1, 0.2]
        gen.add_data(
            pd.DataFrame(
                {"x1": [0.5], "x2": [5.0], "x3": [0.5], "y1": [0.5], "c1": [0.5]}
            )
        )
        with pytest.raises(ValueError):
            gen._get_optimization_bounds()

    @patch.multiple(PatchBayesianGenerator, __abstractmethods__=set())
    def test_fixed_feature(self):
        # test with a fixed feature not contained in vocs
        gen = PatchBayesianGenerator(vocs=TEST_VOCS_BASE)
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
        gen = PatchBayesianGenerator(vocs=TEST_VOCS_BASE)
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
        gen = PatchBayesianGenerator(vocs=TEST_VOCS_BASE)
        gen.fixed_features = {"bad_name": 3.0}
        data = deepcopy(TEST_VOCS_DATA)
        gen.add_data(data)

        with pytest.raises(KeyError):
            gen.train_model()

    @patch.multiple(MultiObjectivePatchBayesianGenerator, __abstractmethods__=set())
    def test_bad_mo_vocs(self):
        vocs = VOCS(
            **{
                "variables": {"x1": [0, 1.0], "x2": [0, 10.0]},
                "objectives": {"y1": "MINIMIZE"},
                "constraints": {"c1": ["GREATER_THAN", 0.5]},
                "constants": {"constant1": 1.0},
            }
        )
        vocs2 = vocs.model_copy()
        vocs2.objectives = {"y1": "MINIMIZE", "y2": "MAXIMIZE"}
        with pytest.raises(VOCSError):
            gen = MultiObjectivePatchBayesianGenerator(
                vocs=vocs, reference_point={"y1": 0.5, "y2": 0.5}
            )

        gen = MultiObjectivePatchBayesianGenerator(
            vocs=vocs2, reference_point={"y1": 0.5, "y2": 0.5}
        )
        assert not gen.supports_single_objective
        with pytest.raises(VOCSError):
            gen.vocs = vocs

    def test_validate_gp_constructor_none(self):
        # Should return StandardModelConstructor instance
        result = BayesianGenerator.validate_gp_constructor(None)
        assert isinstance(result, StandardModelConstructor)

    def test_validate_gp_constructor_instance(self):
        dummy = DummyModelConstructor()
        result = BayesianGenerator.validate_gp_constructor(dummy)
        assert result is dummy

    def test_validate_gp_constructor_str(self):
        result = BayesianGenerator.validate_gp_constructor("standard")
        assert isinstance(result, StandardModelConstructor)
        with pytest.raises(ValueError):
            BayesianGenerator.validate_gp_constructor("not_a_constructor")

    def test_validate_gp_constructor_dict(self):
        # Valid dict
        result = BayesianGenerator.validate_gp_constructor({"name": "standard"})
        assert isinstance(result, StandardModelConstructor)
        # Invalid dict
        with pytest.raises(ValueError):
            BayesianGenerator.validate_gp_constructor({"name": "not_a_constructor"})
