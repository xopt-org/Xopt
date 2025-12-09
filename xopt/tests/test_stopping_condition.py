"""
Test the integration of stopping conditions with Xopt.run()
"""

import pytest
from xopt import (
    Xopt,
    VOCS,
    Evaluator,
    MaxEvaluationsCondition,
    TargetValueCondition,
    CompositeCondition,
    ConvergenceCondition,
)
from xopt.generators.scipy import LatinHypercubeGenerator


class TestStoppingConditionIntegration:
    """Test class for stopping condition integration with Xopt."""

    @pytest.fixture
    def simple_vocs(self):
        """Create a simple VOCS for testing."""
        return VOCS(
            variables={"x1": [0.0, 1.0], "x2": [0.0, 1.0]},
            objectives={"f1": "MINIMIZE"},
        )

    @pytest.fixture
    def test_function(self):
        """Simple quadratic test function."""

        def _test_function(input_dict):
            x1, x2 = input_dict["x1"], input_dict["x2"]
            return {"f1": (x1 - 0.5) ** 2 + (x2 - 0.3) ** 2}

        return _test_function

    @pytest.fixture
    def evaluator(self, test_function):
        """Create evaluator with test function."""
        return Evaluator(function=test_function)

    def test_max_evaluations_condition(self, simple_vocs, evaluator):
        """Test MaxEvaluationsCondition integration."""
        generator = LatinHypercubeGenerator(vocs=simple_vocs)

        X = Xopt(
            vocs=simple_vocs,
            evaluator=evaluator,
            generator=generator,
            stopping_condition=MaxEvaluationsCondition(max_evaluations=5),
        )

        X.run()
        assert len(X.data) == 5

    def test_target_value_condition(self, simple_vocs, evaluator):
        """Test TargetValueCondition integration."""
        generator = LatinHypercubeGenerator(vocs=simple_vocs)

        # Use CompositeCondition to combine target with max evaluations fallback
        stopping_condition = CompositeCondition(
            conditions=[
                TargetValueCondition(
                    objective_name="f1", target_value=0.5, tolerance=0.1
                ),
                MaxEvaluationsCondition(max_evaluations=20),  # Fallback limit
            ],
            logic="or",
        )

        X = Xopt(
            vocs=simple_vocs,
            evaluator=evaluator,
            generator=generator,
            stopping_condition=stopping_condition,
        )

        X.run()

        # Should either reach target or hit max evaluations
        best_value = X.data["f1"].min()
        assert best_value <= 0.6 or len(X.data) == 20

    def test_composite_condition(self, simple_vocs, evaluator):
        """Test CompositeCondition integration."""
        generator = LatinHypercubeGenerator(vocs=simple_vocs)

        composite_condition = CompositeCondition(
            conditions=[
                MaxEvaluationsCondition(max_evaluations=10),
                TargetValueCondition(objective_name="f1", target_value=0.01),
            ],
            logic="or",  # Stop if either condition is met
        )

        X = Xopt(
            vocs=simple_vocs,
            evaluator=evaluator,
            generator=generator,
            stopping_condition=composite_condition,
        )

        X.run()

        # Should stop before or at 10 evaluations
        assert len(X.data) <= 10

    def test_convergence_condition(self, simple_vocs, evaluator):
        """Test ConvergenceCondition integration."""

        # Create data that will show little improvement
        def flat_function(input_dict):
            return {"f1": 1.0}  # Always returns the same value

        flat_evaluator = Evaluator(function=flat_function)
        generator = LatinHypercubeGenerator(vocs=simple_vocs)

        # Use CompositeCondition with max evaluations fallback
        stopping_condition = CompositeCondition(
            conditions=[
                ConvergenceCondition(
                    objective_name="f1", improvement_threshold=0.1, patience=3
                ),
                MaxEvaluationsCondition(max_evaluations=20),  # Fallback limit
            ],
            logic="or",
        )

        X = Xopt(
            vocs=simple_vocs,
            evaluator=flat_evaluator,
            generator=generator,
            stopping_condition=stopping_condition,
        )

        X.run()

        # Should converge quickly due to no improvement
        assert len(X.data) <= 10

    def test_no_stopping_condition_raises_error(self, simple_vocs, evaluator):
        """Test that run() without stopping condition raises ValueError."""
        generator = LatinHypercubeGenerator(vocs=simple_vocs)

        X = Xopt(
            vocs=simple_vocs,
            evaluator=evaluator,
            generator=generator,
            # No stopping_condition
        )

        with pytest.raises(ValueError, match="stopping_condition must be set"):
            X.run()

    def test_composite_condition_fallback(self, simple_vocs, evaluator):
        """Test that CompositeCondition can be used with max evaluations fallback."""
        generator = LatinHypercubeGenerator(vocs=simple_vocs)

        # Create a condition with max evaluations fallback
        stopping_condition = CompositeCondition(
            conditions=[
                TargetValueCondition(
                    objective_name="f1", target_value=-999.0
                ),  # Never reaches
                MaxEvaluationsCondition(max_evaluations=7),  # Should trigger first
            ],
            logic="or",
        )

        X = Xopt(
            vocs=simple_vocs,
            evaluator=evaluator,
            generator=generator,
            stopping_condition=stopping_condition,
        )

        X.run()

        # Should stop at 7 evaluations since target is never reached
        assert len(X.data) == 7

    def test_stopping_condition_from_dict(self, simple_vocs, evaluator):
        """Test creating Xopt with stopping condition from dict."""
        generator = LatinHypercubeGenerator(vocs=simple_vocs)

        # This tests that stopping conditions can be serialized/deserialized
        # For now, we'll create the condition directly since we haven't
        # implemented dict-based creation yet
        condition = MaxEvaluationsCondition(max_evaluations=6)

        X = Xopt(
            vocs=simple_vocs,
            evaluator=evaluator,
            generator=generator,
            stopping_condition=condition,
        )

        X.run()
        assert len(X.data) == 6
