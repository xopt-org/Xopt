from unittest import TestCase

import pandas as pd
import torch
from gpytorch.kernels import PeriodicKernel, ScaleKernel
from xopt.generators.bayesian.expected_improvement import (
    BayesianOptions,
    ExpectedImprovementGenerator,
)

from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.bayesian.options import ModelOptions
from xopt.vocs import VOCS


class TestModelConstructor(StandardModelConstructor):
    def build_model(self, data: pd.DataFrame, tkwargs: dict = None):
        # set tkwargs
        tkwargs = tkwargs or {"dtype": torch.double, "device": "cpu"}

        # drop nans
        valid_data = data[
            pd.unique(self.vocs.variable_names + self.vocs.output_names)
        ].dropna()

        # get data
        input_data, objective_data, constraint_data = self.vocs.extract_data(valid_data)
        train_X = torch.tensor(input_data.to_numpy(), **tkwargs)
        self.input_transform.to(**tkwargs)
        self.likelihood.to(**tkwargs)

        # specify periodic kernel for both objective functions and constraints
        covar_module = ScaleKernel(PeriodicKernel())

        return self.build_standard_model(
            train_X, objective_data, constraint_data, tkwargs, covar_module=covar_module
        )


class TestCustomConstructor(TestCase):
    def test_model_generation(self):
        my_vocs = VOCS(
            variables={"x": [0, 1]},
            objectives={"y": "MAXIMIZE"},
            constraints={"c": ["LESS_THAN", 0]},
        )

        # note the creation of options beforehand
        model_options = ModelOptions(custom_constructor=TestModelConstructor)
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
