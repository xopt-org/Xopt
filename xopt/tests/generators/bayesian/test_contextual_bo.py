import pandas as pd
from xopt import Xopt
from xopt.generators.bayesian import (
    ExpectedImprovementGenerator,
)
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.upper_confidence_bound import (
    UpperConfidenceBoundGenerator,
)
from xopt.vocs import VOCS, ContextualVariable
import pytest
import torch
import numpy as np


class TestContextualBO:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.vocs = VOCS(
            variables={"x1": [0, 1], "x2": ContextualVariable()},
            objectives={"y": "MAXIMIZE"},
        )

        def f(inputs):
            x1 = torch.tensor(inputs["x1"])
            x2 = torch.tensor(0.5)  # fixed contextual variable
            y = torch.sin(2 * 3.14 * x1) * torch.cos(2 * 3.14 * x2)
            return {"y": y, "x2": x2}

        self.evaluator = Evaluator(function=f)

    @pytest.mark.parametrize(
        "generator_class", [ExpectedImprovementGenerator, UpperConfidenceBoundGenerator]
    )
    def test_contextual_ei_generator(self, generator_class):
        generator = generator_class(
            vocs=self.vocs,
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
        # check model output names
        assert generator.model_output_names == ["y"]
        # check model input bounds
        bounds = generator.get_model_input_bounds(data)
        assert bounds["x1"] == [0, 1]
        assert np.allclose(bounds["x2"], [-0.05, 1.05])

        # check training the model
        generator.train_model(generator.data)

        # check optimization bounds - should be 1D for x1 only
        opt_bounds = generator._get_optimization_bounds()
        assert torch.allclose(opt_bounds, torch.tensor([[0.0], [1.0]]).double())

        generator.generate(1)

    @pytest.mark.parametrize(
        "generator_class",
        [ExpectedImprovementGenerator, UpperConfidenceBoundGenerator],
    )
    def test_xopt_step_with_contextual_variable(self, generator_class):
        generator = generator_class(vocs=self.vocs)
        xopt = Xopt(generator=generator, evaluator=self.evaluator, vocs=self.vocs)

        # Seed initial data without providing the contextual variable as an input.
        xopt.evaluate_data(pd.DataFrame({"x1": np.linspace(0, 1, 5)}))

        # Generated candidates should only include controllable variables.
        candidates = xopt.generator.generate(1)
        if isinstance(candidates, list):
            assert all("x2" not in candidate for candidate in candidates)
        else:
            assert "x2" not in candidates.columns

        # Evaluation should succeed without requiring contextual inputs.
        xopt.evaluate_data(candidates)
        assert "x2" in xopt.data.columns

    def test_visualize_model_with_contextual_axis_warns(self):
        generator = UpperConfidenceBoundGenerator(vocs=self.vocs)
        xopt = Xopt(generator=generator, evaluator=self.evaluator)
        xopt.evaluate_data(pd.DataFrame({"x1": np.linspace(0, 1, 5)}))
        xopt.generator.train_model()

        with pytest.warns(RuntimeWarning, match="Acquisition plot unavailable"):
            xopt.generator.visualize_model(
                output_names=["y"],
                variable_names=["x1", "x2"],
                show_acquisition=True,
                n_grid=5,
            )
