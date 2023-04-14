import json
import os

import pandas as pd
import torch
import yaml
from gpytorch.kernels import PeriodicKernel, ScaleKernel

from xopt.generators import ExpectedImprovementGenerator
from xopt.generators.bayesian.options import (
    AcqOptions,
    BayesianOptions,
    ModelOptions,
    OptimOptions,
)
from xopt.vocs import VOCS


class TestBayesianOptions:
    def test_default(self):
        AcqOptions()
        OptimOptions()
        model_options = ModelOptions()
        BayesianOptions()
        model_options.json()

    def test_json_serialization(self):
        options = BayesianOptions()
        options.json()

    def test_model_saving(self):
        my_vocs = VOCS(
            variables={"x": [0, 1]},
            objectives={"y": "MAXIMIZE"},
            constraints={"c": ["LESS_THAN", 0]},
        )

        # specify a periodic kernel for each output (objectives and constraints)
        covar_module = ScaleKernel(PeriodicKernel())

        model_options = ModelOptions(covar_modules=covar_module)
        generator_options = BayesianOptions(model=model_options)
        generator = ExpectedImprovementGenerator(my_vocs, options=generator_options)

        # define training data to pass to the generator
        train_x = torch.tensor((0.2, 0.5, 0.6))
        train_y = 5.0 * torch.cos(2 * 3.14 * train_x + 0.25)
        train_c = 2.0 * torch.sin(2 * 3.14 * train_x + 0.25)

        training_data = pd.DataFrame(
            {"x": train_x.numpy(), "y": train_y.numpy(), "c": train_c}
        )

        generator.add_data(training_data)

        # save generator config to file
        options = json.loads(generator.options.json())

        with open("test.yml", "w") as f:
            yaml.dump(options, f)

        # load generator config from file
        with open("test.yml", "r") as f:
            saved_options = yaml.safe_load(f)

        # create generator from file
        loaded_generator_options = BayesianOptions(**saved_options)
        assert isinstance(loaded_generator_options.model.covar_modules, ScaleKernel)

        # clean up
        os.remove("test.yml")
        os.remove(options["model"]["covar_modules"])

        # specify a periodic kernel for each output (objectives and constraints)
        covar_modules = {
            "y": ScaleKernel(PeriodicKernel()),
            "c": ScaleKernel(PeriodicKernel()),
        }

        model_options = ModelOptions(covar_modules=covar_modules)
        generator_options = BayesianOptions(model=model_options)
        generator = ExpectedImprovementGenerator(my_vocs, options=generator_options)

        # define training data to pass to the generator
        train_x = torch.tensor((0.2, 0.5, 0.6))
        train_y = 5.0 * torch.cos(2 * 3.14 * train_x + 0.25)
        train_c = 2.0 * torch.sin(2 * 3.14 * train_x + 0.25)

        training_data = pd.DataFrame(
            {"x": train_x.numpy(), "y": train_y.numpy(), "c": train_c}
        )

        generator.add_data(training_data)

        # save generator config to file
        options = json.loads(generator.options.json())

        with open("test.yml", "w") as f:
            yaml.dump(options, f)

        # load generator config from file
        with open("test.yml", "r") as f:
            saved_options = yaml.safe_load(f)

        # create generator from file
        loaded_generator_options = BayesianOptions(**saved_options)
        for name, val in loaded_generator_options.model.covar_modules.items():
            assert isinstance(val, ScaleKernel)

        # clean up
        os.remove("test.yml")
        for name in my_vocs.output_names:
            os.remove(options["model"]["covar_modules"][name])
