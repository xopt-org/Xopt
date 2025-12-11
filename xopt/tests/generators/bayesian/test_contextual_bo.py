import pandas as pd
from xopt.generators.bayesian.contextual_ei import (
    ContextualExpectedImprovementGenerator,
)
from xopt.evaluator import Evaluator
from xopt.vocs import VOCS
import pytest
import torch
import numpy as np


class TestContextualBO:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.vocs = VOCS(
            variables={"x1": [0, 1]},
            objectives={"y": "MAXIMIZE"},
            observables=["x2"],
        )

        def f(inputs):
            x1 = inputs["x1"]
            x2 = torch.tensor(0.5)  # fixed contextual variable
            y = torch.sin(2 * 3.14 * x1) * torch.cos(2 * 3.14 * x2)
            return {"y": y, "x2": x2}

        self.evaluator = Evaluator(function=f)

    def test_contextual_ei_generator(self):
        generator = ContextualExpectedImprovementGenerator(
            vocs=self.vocs,
            contextual_observables=["x2"],
        )

        # Create initial data with varying x2
        x1 = np.linspace(0, 1, 5)
        x2 = np.linspace(0, 1, 5)  # varying contextual variable
        y = np.sin(2 * 3.14 * x1) * np.cos(2 * 3.14 * x2)
        initial_data = {"x1": x1, "x2": x2, "y": y}

        data = pd.DataFrame(initial_data)
        generator.add_data(data)

        # check model input names
        assert generator.model_input_names == ["x1", "x2"]
        # check model input bounds
        bounds = generator.get_model_input_bounds(data)
        assert bounds["x1"] == (0, 1)
        assert np.allclose(bounds["x2"], (0.0, 1.0 + 1e-6))

        # check training the model
        generator.train_model(generator.data)

        # check optimization bounds - should be 1D for x1 only
        opt_bounds = generator._get_optimization_bounds()
        assert torch.allclose(opt_bounds, torch.tensor([[0.0], [1.0]]).double())

        generator.generate(1)
