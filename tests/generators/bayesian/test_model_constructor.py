import json
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from gpytorch.kernels import PeriodicKernel, ScaleKernel
from pydantic import ValidationError

from xopt.generators.bayesian.expected_improvement import ExpectedImprovementGenerator
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA
from xopt.vocs import VOCS


class TestModelConstructor:
    def test_standard(self):
        test_data = deepcopy(TEST_VOCS_DATA)
        test_vocs = deepcopy(TEST_VOCS_BASE)

        constructor = StandardModelConstructor()

        constructor.build_model(
            test_vocs.variable_names, test_vocs.output_names, test_data
        )

        constructor.build_model_from_vocs(test_vocs, test_data)

    def test_custom_model(self):
        test_data = deepcopy(TEST_VOCS_DATA)
        test_vocs = deepcopy(TEST_VOCS_BASE)

        custom_covar = {"y1": ScaleKernel(PeriodicKernel())}

        with pytest.raises(ValidationError):
            StandardModelConstructor(
                vocs=test_vocs, covar_modules=deepcopy(custom_covar)["y1"]
            )

        # test custom covar module
        constructor = StandardModelConstructor(covar_modules=deepcopy(custom_covar))
        model = constructor.build_model(
            test_vocs.variable_names, test_vocs.output_names, test_data
        )
        assert isinstance(model.models[0].covar_module.base_kernel, PeriodicKernel)

        # test prior mean
        class ConstraintPrior(torch.nn.Module):
            def forward(self, X):
                return X[:, 0] ** 2

        mean_modules = {"c1": ConstraintPrior()}
        constructor = StandardModelConstructor(mean_modules=mean_modules)
        model = constructor.build_model_from_vocs(test_vocs, test_data)
        assert isinstance(model.models[1].mean_module.model, ConstraintPrior)

    def test_model_w_nans(self):
        test_data = deepcopy(TEST_VOCS_DATA)
        test_vocs = deepcopy(TEST_VOCS_BASE)
        constructor = StandardModelConstructor()

        # add nans to ouputs
        test_data.loc[5, "y1"] = np.nan
        test_data.loc[6, "c1"] = np.nan
        test_data.loc[7, "c1"] = np.nan

        model = constructor.build_model_from_vocs(test_vocs, test_data)

        assert model.train_inputs[0][0].shape == torch.Size([9, 2])
        assert model.train_inputs[1][0].shape == torch.Size([8, 2])

        # add nans to inputs
        test_data2 = deepcopy(TEST_VOCS_DATA)
        test_data2.loc[5, "x1"] = np.nan

        model2 = constructor.build_model_from_vocs(test_vocs, test_data2)
        assert model2.train_inputs[0][0].shape == torch.Size([9, 2])

        # add nans to both
        test_data3 = deepcopy(TEST_VOCS_DATA)
        test_data3.loc[5, "x1"] = np.nan
        test_data3.loc[7, "c1"] = np.nan

        model3 = constructor.build_model_from_vocs(test_vocs, test_data3)
        assert model3.train_inputs[0][0].shape == torch.Size([9, 2])
        assert model3.train_inputs[1][0].shape == torch.Size([8, 2])

    def test_serialization(self):
        # test custom covar module
        custom_covar = {"y1": ScaleKernel(PeriodicKernel())}
        constructor = StandardModelConstructor(covar_modules=custom_covar)
        constructor.json()

        import os

        os.remove("covar_modules_y1.pt")

    def test_model_saving(self):
        my_vocs = VOCS(
            variables={"x": [0, 1]},
            objectives={"y": "MAXIMIZE"},
            constraints={"c": ["LESS_THAN", 0]},
        )

        # specify a periodic kernel for each output (objectives and constraints)
        covar_modules = {"y": ScaleKernel(PeriodicKernel())}

        model_constructor = StandardModelConstructor(covar_modules=covar_modules)
        generator = ExpectedImprovementGenerator(
            vocs=my_vocs, model_constructor=model_constructor
        )

        # define training data to pass to the generator
        train_x = torch.tensor((0.2, 0.5, 0.6))
        train_y = 5.0 * torch.cos(2 * 3.14 * train_x + 0.25)
        train_c = 2.0 * torch.sin(2 * 3.14 * train_x + 0.25)

        training_data = pd.DataFrame(
            {"x": train_x.numpy(), "y": train_y.numpy(), "c": train_c}
        )

        generator.add_data(training_data)

        # save generator config to file
        options = json.loads(generator.json())

        with open("test.yml", "w") as f:
            yaml.dump(options, f)

        # load generator config from file
        with open("test.yml", "r") as f:
            saved_options_dict = yaml.safe_load(f)

        # create generator from dict
        saved_options_dict["vocs"] = my_vocs.dict()
        loaded_generator = ExpectedImprovementGenerator.parse_raw(
            json.dumps(saved_options_dict)
        )
        assert isinstance(
            loaded_generator.model_constructor.covar_modules["y"], ScaleKernel
        )

        # clean up
        os.remove("test.yml")
        os.remove(options["model_constructor"]["covar_modules"]["y"])

        # specify a periodic kernel for each output (objectives and constraints)
        covar_modules = {
            "y": ScaleKernel(PeriodicKernel()),
            "c": ScaleKernel(PeriodicKernel()),
        }

        model_constructor = StandardModelConstructor(covar_modules=covar_modules)
        generator = ExpectedImprovementGenerator(
            vocs=my_vocs, model_constructor=model_constructor
        )

        # define training data to pass to the generator
        train_x = torch.tensor((0.2, 0.5, 0.6))
        train_y = 5.0 * torch.cos(2 * 3.14 * train_x + 0.25)
        train_c = 2.0 * torch.sin(2 * 3.14 * train_x + 0.25)

        training_data = pd.DataFrame(
            {"x": train_x.numpy(), "y": train_y.numpy(), "c": train_c}
        )

        generator.add_data(training_data)

        # save generator config to file
        options = json.loads(generator.json())

        with open("test.yml", "w") as f:
            yaml.dump(options, f)

        # load generator config from file
        with open("test.yml", "r") as f:
            saved_options = yaml.safe_load(f)

        # create generator from file
        saved_options["vocs"] = my_vocs.dict()
        loaded_generator = ExpectedImprovementGenerator.parse_raw(
            json.dumps(saved_options)
        )
        for name, val in loaded_generator.model_constructor.covar_modules.items():
            assert isinstance(val, ScaleKernel)

        # clean up
        os.remove("test.yml")
        for name in my_vocs.output_names:
            os.remove(options["model_constructor"]["covar_modules"][name])
