from unittest import TestCase

import pandas as pd
import torch
from gpytorch.kernels import PeriodicKernel, ScaleKernel

from xopt.generators.bayesian.expected_improvement import (
    BayesianOptions,
    ExpectedImprovementGenerator,
)
from xopt.generators.bayesian.options import ModelOptions
from xopt.vocs import VOCS


class TestCustomConstructor(TestCase):
    def test_uniform_model_generation(self):
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
        generator.train_model()

    def test_nonuniform_model_generation(self):
        my_vocs = VOCS(
            variables={"x": [0, 1]},
            objectives={"y": "MAXIMIZE"},
            constraints={"c": ["LESS_THAN", 0]},
        )

        # specify a periodic kernel for each output (objectives and constraints)
        covar_module = {"y1": ScaleKernel(PeriodicKernel())}

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
        generator.train_model()

    def test_prior_mean(self):
        my_vocs = VOCS(
            variables={"x": [0, 1]},
            objectives={"y": "MAXIMIZE"},
            constraints={"c": ["LESS_THAN", 0]},
        )

        class ConstraintPrior(torch.nn.Module):
            def forward(self, X):
                return X.squeeze(dim=-1) ** 2

        model_options = ModelOptions(mean_modules={"c": ConstraintPrior()})
        generator_options = BayesianOptions(model=model_options)
        generator = ExpectedImprovementGenerator(my_vocs, options=generator_options)

        # define training data to pass to the generator
        train_x = torch.tensor((0.2, 0.5))
        train_y = 1.0 * torch.cos(2 * 3.14 * train_x + 0.25)
        train_c = train_x**2

        training_data = pd.DataFrame(
            {"x": train_x.numpy(), "y": train_y.numpy(), "c": train_c}
        )

        generator.add_data(training_data)
        generator.train_model()
