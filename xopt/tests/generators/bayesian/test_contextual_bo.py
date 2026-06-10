import pandas as pd
from xopt import Xopt
from xopt.generators.bayesian import (
    ExpectedImprovementGenerator,
)
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.upper_confidence_bound import (
    UpperConfidenceBoundGenerator,
)
from xopt.vocs import VOCS, ContextualVariable, convert_dataframe_to_inputs
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
        assert bounds["x1"] == (0.0, 1.0)
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

    def test_get_optimum_with_contextual_variable(self):
        """Test that get_optimum works correctly when contextual variables are present."""
        generator = UpperConfidenceBoundGenerator(vocs=self.vocs)

        x1 = np.linspace(0, 1, 5)
        x2 = np.linspace(0.2, 0.8, 5)
        y = np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2)
        data = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        generator.add_data(data)
        generator.train_model()

        optimum = generator.get_optimum()
        # optimum should only contain x1 (the controllable variable)
        assert "x1" in optimum.columns
        assert optimum["x1"].iloc[0] >= 0.0
        assert optimum["x1"].iloc[0] <= 1.0

    def test_contextual_variable_bounds_from_data(self):
        """Test that model input bounds for contextual variables are derived from data."""
        generator = UpperConfidenceBoundGenerator(vocs=self.vocs)
        x1 = np.linspace(0, 1, 5)
        x2 = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        y = np.ones(5)
        data = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        generator.add_data(data)

        bounds = generator.get_model_input_bounds(data)
        # x2 bounds should be derived from data with 5% padding
        width = 0.7 - 0.3
        padding = 0.05 * width
        assert np.isclose(bounds["x2"][0], 0.3 - padding)
        assert np.isclose(bounds["x2"][1], 0.7 + padding)

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

    def test_candidate_names_excludes_contextual_and_fixed(self):
        """_candidate_names must exclude both fixed features and contextual variables."""
        # VOCS: x1 (controllable), x2 (contextual), x3 (controllable)
        vocs = VOCS(
            variables={"x1": [0, 1], "x2": ContextualVariable(), "x3": [0, 1]},
            objectives={"y": "MAXIMIZE"},
        )
        generator = UpperConfidenceBoundGenerator(vocs=vocs)

        # Without fixed features: only contextual x2 excluded
        assert generator._candidate_names == ["x1", "x3"]

        # With x3 fixed: both x2 (contextual) and x3 (fixed) excluded
        generator.fixed_features = {"x3": 0.5}
        assert generator._candidate_names == ["x1"]

        # Calling _candidate_names multiple times must not mutate vocs
        _ = generator._candidate_names
        assert generator.vocs.variable_names == ["x1", "x2", "x3"]

    def test_optimization_bounds_excludes_contextual_and_fixed(self):
        """_get_optimization_bounds must correctly exclude both fixed features and
        contextual variables in a single pass, avoiding double-slicing."""
        # VOCS: x1 [0,1], x2 contextual, x3 [0,2]
        vocs = VOCS(
            variables={"x1": [0, 1], "x2": ContextualVariable(), "x3": [0, 2]},
            objectives={"y": "MAXIMIZE"},
        )
        generator = UpperConfidenceBoundGenerator(vocs=vocs)

        # With x3 fixed: only x1 bounds should remain
        generator.fixed_features = {"x3": 1.0}
        bounds = generator._get_optimization_bounds()
        assert bounds.shape == (2, 1)
        assert torch.allclose(bounds, torch.tensor([[0.0], [1.0]]).double())

    def test_convert_dataframe_to_inputs_error_message_contextual(self):
        """The error message when input columns mismatch must mention non-contextual
        variables so users are not confused by contextual variables being absent."""
        vocs = VOCS(
            variables={"x1": [0, 1], "x2": ContextualVariable()},
            objectives={"y": "MAXIMIZE"},
        )
        # Providing x2 (contextual) instead of x1 should raise with an informative message
        with pytest.raises(
            ValueError,
            match="non-contextual",
        ):
            convert_dataframe_to_inputs(vocs, pd.DataFrame({"x2": [0.5]}))
