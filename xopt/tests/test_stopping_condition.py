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

        X = Xopt(
            vocs=simple_vocs,
            evaluator=evaluator,
            generator=generator,
            stopping_condition=TargetValueCondition(
                objective_name="f1", target_value=0.5, tolerance=0.1
            ),
            max_evaluations=20,  # Fallback limit
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

        X = Xopt(
            vocs=simple_vocs,
            evaluator=flat_evaluator,
            generator=generator,
            stopping_condition=ConvergenceCondition(
                objective_name="f1", improvement_threshold=0.1, patience=3
            ),
            max_evaluations=20,
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
            # No stopping_condition or max_evaluations
        )

        with pytest.raises(
            ValueError, match="Either stopping_condition or max_evaluations must be set"
        ):
            X.run()

    def test_max_evaluations_as_fallback(self, simple_vocs, evaluator):
        """Test that max_evaluations can be used as fallback when stopping condition never triggers."""
        generator = LatinHypercubeGenerator(vocs=simple_vocs)

        # Create a condition that will never trigger
        never_stop_condition = TargetValueCondition(
            objective_name="f1",
            target_value=-999.0,  # Impossible to reach
        )

        X = Xopt(
            vocs=simple_vocs,
            evaluator=evaluator,
            generator=generator,
            stopping_condition=never_stop_condition,
            max_evaluations=7,
        )

        X.run()

        # Should stop at max_evaluations since condition never triggers
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

    def test_stopping_condition_priority_over_max_evaluations(
        self, simple_vocs, evaluator
    ):
        """Test that stopping condition takes priority over max_evaluations."""
        generator = LatinHypercubeGenerator(vocs=simple_vocs)

        X = Xopt(
            vocs=simple_vocs,
            evaluator=evaluator,
            generator=generator,
            stopping_condition=MaxEvaluationsCondition(max_evaluations=3),
            max_evaluations=10,  # Higher fallback limit
        )

        X.run()

        # Should stop at 3 (from stopping condition), not 10 (from max_evaluations)
        assert len(X.data) == 3
