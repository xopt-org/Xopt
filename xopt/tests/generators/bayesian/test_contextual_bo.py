import pandas as pd
from xopt import Xopt
from xopt.generators.bayesian import (
    ExpectedImprovementGenerator,
)
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.upper_confidence_bound import (
    UpperConfidenceBoundGenerator,
)
from xopt.generators.bayesian.turbo import OptimizeTurboController
from xopt.generators.bayesian.utils import validate_turbo_controller_center
from xopt.errors import VOCSError
from xopt.vocs import VOCS, ContextualVariable, convert_dataframe_to_inputs
import pytest
import torch
import numpy as np


def _contextual_eval_for_yaml_roundtrip(inputs):
    pass # pragma: no cover


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
        assert all("x2" not in candidate for candidate in candidates)


        # Evaluation should succeed without requiring contextual inputs.
        xopt.evaluate_data(candidates)
        assert "x2" in xopt.data.columns

    def test_xopt_yaml_roundtrip_with_contextual_variable(self):
        vocs = VOCS(
            variables={"x1": [0, 1], "x2": ContextualVariable()},
            objectives={"y": "MAXIMIZE"},
        )
        xopt = Xopt(
            generator=UpperConfidenceBoundGenerator(vocs=vocs),
            evaluator=Evaluator(function=_contextual_eval_for_yaml_roundtrip),
        )

        reloaded = Xopt.from_yaml(xopt.yaml())

        assert isinstance(reloaded.vocs.variables["x2"], ContextualVariable)

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

    def test_contextual_variable_explicit_domain_overrides_data(self):
        vocs = VOCS(
            variables={"x1": [0, 1], "x2": ContextualVariable(domain=[0.2, 0.4])},
            objectives={"y": "MAXIMIZE"},
        )
        generator = UpperConfidenceBoundGenerator(vocs=vocs)

        data = pd.DataFrame(
            {
                "x1": np.linspace(0, 1, 5),
                "x2": np.array([0.3, 0.4, 0.5, 0.6, 0.7]),
                "y": np.ones(5),
            }
        )
        bounds = generator.get_model_input_bounds(data)
        assert np.allclose(bounds["x2"], [0.2, 0.4])

    def test_contextual_variable_explicit_domain_does_not_require_data_column(self):
        vocs = VOCS(
            variables={"x1": [0, 1], "x2": ContextualVariable(domain=[-1.0, 1.0])},
            objectives={"y": "MAXIMIZE"},
        )
        generator = UpperConfidenceBoundGenerator(vocs=vocs)

        data = pd.DataFrame({"x1": np.linspace(0, 1, 5), "y": np.ones(5)})
        bounds = generator.get_model_input_bounds(data)
        assert np.allclose(bounds["x2"], [-1.0, 1.0])

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

    @pytest.mark.parametrize(
        "generator_class",
        [ExpectedImprovementGenerator, UpperConfidenceBoundGenerator],
    )
    def test_generate_with_contextual_and_fixed_features(self, generator_class):
        """Generated candidates should omit contextual variables and respect fixed
        feature values."""
        vocs = VOCS(
            variables={"x1": [0, 1], "x2": ContextualVariable(), "x3": [0, 1]},
            objectives={"y": "MAXIMIZE"},
        )
        generator = generator_class(vocs=vocs, fixed_features={"x3": 0.25})

        data = pd.DataFrame(
            {
                "x1": np.linspace(0.0, 1.0, 8),
                "x2": np.linspace(0.2, 0.8, 8),
                "x3": np.linspace(0.0, 1.0, 8),
                "y": np.sin(2 * np.pi * np.linspace(0.0, 1.0, 8)),
            }
        )
        generator.add_data(data)
        generator.train_model()

        candidates = generator.generate(2)
        assert all("x1" in candidate for candidate in candidates)
        assert all("x2" not in candidate for candidate in candidates)
        assert all("x3" in candidate for candidate in candidates)
        assert all(np.isclose(candidate["x3"], 0.25) for candidate in candidates)
        assert all(0.0 <= candidate["x1"] <= 1.0 for candidate in candidates)


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

    def test_contextual_variable_with_turbo_controller(self):
        """Contextual variables and TuRBO should work together, with TuRBO operating
        over controllable variables only."""
        vocs = VOCS(
            variables={"x1": [0, 1], "x2": ContextualVariable()},
            objectives={"y": "MAXIMIZE"},
        )
        turbo = OptimizeTurboController(vocs=vocs)
        generator = UpperConfidenceBoundGenerator(vocs=vocs, turbo_controller=turbo)

        data = pd.DataFrame(
            {
                "x1": np.linspace(0.0, 1.0, 8),
                "x2": np.linspace(0.2, 0.8, 8),
                "y": np.sin(2 * np.pi * np.linspace(0.0, 1.0, 8)),
            }
        )
        generator.add_data(data)
        generator.train_model()

        # TuRBO trust region should stay in controllable dimensions only.
        tr = generator.turbo_controller.get_trust_region(generator)
        assert tr.shape == (2, 1)

        # Optimization bounds should likewise be 1D (x1 only).
        opt_bounds = generator._get_optimization_bounds()
        assert opt_bounds.shape == (2, 1)

        # Candidate generation should exclude contextual variables.
        candidates = generator.generate(1)
        assert all("x2" not in candidate for candidate in candidates)

    def test_bayesian_generator_contextual_support_flag(self):
        """Bayesian generators must reject contextual variables when support is disabled."""

        class NoContextUCBGenerator(UpperConfidenceBoundGenerator):
            supports_contextual_variables: bool = False

        vocs = VOCS(
            variables={"x1": [0, 1], "x2": ContextualVariable()},
            objectives={"y": "MAXIMIZE"},
        )

        with pytest.raises(
            VOCSError, match="this generator does not support contextual variables"
        ):
            NoContextUCBGenerator(vocs=vocs)

    def test_turbo_fallback_variable_names_exclude_contextual(self):
        """TuRBO should fall back to non-contextual VOCS variables when candidate
        names are unavailable."""
        vocs = VOCS(
            variables={"x1": [0, 1], "x2": ContextualVariable(), "x3": [-1, 1]},
            objectives={"y": "MAXIMIZE"},
        )
        turbo = OptimizeTurboController(vocs=vocs)

        dummy_generator = type("DummyGenerator", (), {})()
        dummy_generator.vocs = vocs

        assert turbo._get_trust_region_variable_names(dummy_generator) == ["x1", "x3"]

    def test_turbo_center_x_with_contextual_and_fixed_features(self):
        """TuRBO center validation and data filtering should operate over active
        candidate dimensions only."""
        vocs = VOCS(
            variables={"x1": [0, 1], "x2": ContextualVariable(), "x3": [0, 1]},
            objectives={"y": "MAXIMIZE"},
        )
        turbo = OptimizeTurboController(
            vocs=vocs,
            center_x={"x1": 0.5, "x2": 10.0, "x3": 10.0},
            length=0.5,
        )
        generator = UpperConfidenceBoundGenerator(
            vocs=vocs,
            fixed_features={"x3": 0.25},
            turbo_controller=turbo,
        )

        data = pd.DataFrame(
            {
                "x1": [0.1, 0.4, 0.6, 0.9],
                "x2": [0.2, 0.4, 0.6, 0.8],
                "x3": [0.25, 0.25, 0.25, 0.25],
                "y": [1.0, 2.0, 3.0, 4.0],
            }
        )

        filtered = generator.get_training_data(data)
        assert not filtered.empty
        assert filtered["x1"].between(0.25, 0.75).all()

    def test_turbo_center_validation_skips_missing_candidate_key(self):
        """Center validation should skip candidate dimensions absent from center_x."""
        vocs = VOCS(
            variables={"x1": [0, 1], "x2": [0, 1]},
            objectives={"y": "MAXIMIZE"},
        )

        class DummyTurboController(OptimizeTurboController):
            def get_trust_region(self, generator):
                return torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        turbo = DummyTurboController(
            vocs=vocs,
            center_x={"x1": 0.5, "x2": 0.5},
        )

        class DummyGenerator:
            def __init__(self, vocs, turbo_controller):
                self.vocs = vocs
                self.turbo_controller = turbo_controller

            @property
            def _candidate_names(self):
                return ["x1", "missing_feature"]

        generator = DummyGenerator(vocs, turbo)
        validate_turbo_controller_center(generator)
