"""
Test the integration of stopping conditions with Xopt.run()
"""

import json

import pandas as pd
import pytest

from xopt import (
    VOCS,
    Evaluator,
    Xopt,
)
from xopt.generators.scipy import LatinHypercubeGenerator
from xopt.generators.sequential.neldermead import NelderMeadGenerator
from xopt.stopping_conditions import (
    CompositeCondition,
    ConvergenceCondition,
    FeasibilityCondition,
    MaxEvaluationsCondition,
    StagnationCondition,
    TargetValueCondition,
    get_stopping_condition,
)


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

    def test_max_evaluation(self, simple_vocs, evaluator):
        """test with explict dataset max evaluations"""
        data = pd.DataFrame(
            {
                "x1": [0.1, 0.2, 0.3],
                "x2": [0.1, 0.2, 0.3],
                "f1": [0.5, 0.4, 0.3],
            }
        )
        condition = MaxEvaluationsCondition(max_evaluations=5)
        assert not condition.should_stop(data, simple_vocs)

        condition = MaxEvaluationsCondition(max_evaluations=2)
        assert condition.should_stop(data, simple_vocs)

        # test cases where count_valid_only and use_dataframe_index are True
        condition = MaxEvaluationsCondition(max_evaluations=3, count_valid_only=True)
        data_with_errors = data.copy()
        data_with_errors.loc[1, "error"] = "true"  # Simulate an error
        data_with_errors.loc[2, "error"] = "true"  # Simulate an error

        assert not condition.should_stop(data_with_errors, simple_vocs)

        # test case where use_dataframe_index is True
        condition = MaxEvaluationsCondition(max_evaluations=5, use_dataframe_index=True)
        data_indexed = data.copy()
        data_indexed.index = [0, 2, 7]  # Simulate sparse index
        assert condition.should_stop(data_indexed, simple_vocs)

    def test_target_value(self, simple_vocs, evaluator):
        """test with explict dataset target value"""
        data = pd.DataFrame(
            {
                "x1": [0.1, 0.2, 0.3],
                "x2": [0.1, 0.2, 0.3],
                "f1": [0.5, 0.4, 0.3],
            }
        )
        condition = TargetValueCondition(
            objective_name="f1", target_value=0.25, tolerance=0.01
        )
        assert not condition.should_stop(data, simple_vocs)

        condition = TargetValueCondition(
            objective_name="f1", target_value=0.35, tolerance=0.01
        )
        assert condition.should_stop(data, simple_vocs)

        vocs_max = VOCS(
            variables=simple_vocs.variables,
            objectives={"f1": "MAXIMIZE"},
        )
        condition = TargetValueCondition(
            objective_name="f1", target_value=0.25, tolerance=0.01
        )
        assert condition.should_stop(data, vocs_max)

        condition = TargetValueCondition(
            objective_name="f1", target_value=0.65, tolerance=0.01
        )
        assert not condition.should_stop(data, vocs_max)

        # test with no data
        empty_data = pd.DataFrame()
        condition = TargetValueCondition(
            objective_name="f1", target_value=0.35, tolerance=0.01
        )
        assert not condition.should_stop(empty_data, simple_vocs)

        # test with nan values
        nan_data = data.copy()
        nan_data.loc[:, "f1"] = float("nan")
        assert not condition.should_stop(nan_data, simple_vocs)

    def test_convergence(self, simple_vocs, evaluator):
        """test with explict dataset convergence"""
        data = pd.DataFrame(
            {
                "x1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "x2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "f1": [0.5, 0.4, 0.35, 0.34, 0.339, 0.338],
            }
        )
        condition = ConvergenceCondition(
            objective_name="f1", improvement_threshold=0.01, patience=2
        )
        assert condition.should_stop(data, simple_vocs)

        # test with relative improvement threshold
        condition = ConvergenceCondition(
            objective_name="f1", improvement_threshold=0.05, patience=2, relative=True
        )
        assert condition.should_stop(data, simple_vocs)

        condition = ConvergenceCondition(
            objective_name="f1", improvement_threshold=0.001, patience=2
        )
        assert not condition.should_stop(data, simple_vocs)

        # test with fewer data points than patience
        condition = ConvergenceCondition(
            objective_name="f1", improvement_threshold=0.01, patience=5
        )
        assert not condition.should_stop(data.iloc[:4], simple_vocs)

        # test with maximize objective
        vocs_max = VOCS(
            variables=simple_vocs.variables,
            objectives={"f1": "MAXIMIZE"},
        )
        condition = ConvergenceCondition(
            objective_name="f1", improvement_threshold=0.01, patience=2
        )
        max_data = data.copy()
        max_data["f1"] = -max_data["f1"]  # Invert for maximization
        assert condition.should_stop(max_data, vocs_max)

    def test_stagnation(self, simple_vocs, evaluator):
        """test with explict dataset stagnation"""
        data = pd.DataFrame(
            {
                "x1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "x2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "f1": [0.5, 0.4, 0.35, 0.34, 0.35, 0.5],
            }
        )
        condition = StagnationCondition(objective_name="f1", patience=2, tolerance=0.01)
        assert condition.should_stop(data, simple_vocs)

        condition = StagnationCondition(objective_name="f1", patience=2, tolerance=0.01)
        assert not condition.should_stop(data.iloc[:4], simple_vocs)

        # test with fewer data points than patience
        condition = StagnationCondition(objective_name="f1", patience=5, tolerance=0.01)
        assert not condition.should_stop(data.iloc[:4], simple_vocs)

        # test with maximize objective
        vocs_max = VOCS(
            variables=simple_vocs.variables,
            objectives={"f1": "MAXIMIZE"},
        )
        condition = StagnationCondition(objective_name="f1", patience=2, tolerance=0.01)
        max_data = data.copy()
        max_data["f1"] = -max_data["f1"]  # Invert for maximization
        assert condition.should_stop(max_data, vocs_max)

        # test with linearly improving data
        improving_data = pd.DataFrame(
            {
                "x1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "x2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "f1": [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            }
        )
        condition = StagnationCondition(objective_name="f1", patience=2, tolerance=0.01)
        assert not condition.should_stop(improving_data, simple_vocs)

    def test_composite(self, simple_vocs, evaluator):
        """test with explict dataset composite condition"""
        data = pd.DataFrame(
            {
                "x1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "x2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "f1": [0.5, 0.4, 0.35, 0.34, 0.339, 0.338],
            }
        )
        condition = CompositeCondition(
            conditions=[
                MaxEvaluationsCondition(max_evaluations=10),
                TargetValueCondition(
                    objective_name="f1", target_value=0.4, tolerance=0.01
                ),
            ],
            logic="or",
        )
        assert condition.should_stop(data, simple_vocs)

        condition = CompositeCondition(
            conditions=[
                MaxEvaluationsCondition(max_evaluations=5),
                TargetValueCondition(
                    objective_name="f1", target_value=0.25, tolerance=0.01
                ),
            ],
            logic="or",
        )
        assert condition.should_stop(data, simple_vocs)

    @pytest.mark.parametrize(
        "condition",
        [
            MaxEvaluationsCondition(max_evaluations=2),
            TargetValueCondition(objective_name="f1", target_value=10.1, tolerance=0.1),
            ConvergenceCondition(
                objective_name="f1", improvement_threshold=10.0, patience=5
            ),
            StagnationCondition(objective_name="f1", patience=5, tolerance=10.0),
            CompositeCondition(
                conditions=[
                    MaxEvaluationsCondition(max_evaluations=2),
                    TargetValueCondition(
                        objective_name="f1", target_value=10.1, tolerance=0.1
                    ),
                ],
                logic="or",
            ),
        ],
        ids=[
            "max_evaluations",
            "target_value",
            "convergence",
            "stagnation",
            "composite_or",
        ],
    )
    def test_conditions_in_xopt(self, simple_vocs, evaluator, condition):
        """Test each stopping condition individually."""
        generator = NelderMeadGenerator(vocs=simple_vocs)

        X = Xopt(
            vocs=simple_vocs,
            evaluator=evaluator,
            generator=generator,
            stopping_condition=condition,
        )
        X.random_evaluate(1)
        X.run()

        # Basic assertions depending on condition type
        if isinstance(condition, MaxEvaluationsCondition):
            assert len(X.data) == condition.max_evaluations

        # test round trip serialization
        info = json.loads(X.json())
        info["evaluator"] = evaluator
        Xopt(**info)

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

    def test_feasibility_condition(self, evaluator):
        """Test FeasibilityCondition."""
        # Create VOCS with constraints
        constrained_vocs = VOCS(
            variables={"x1": [0.0, 1.0], "x2": [0.0, 1.0]},
            objectives={"f1": "MINIMIZE"},
            constraints={"c1": ["GREATER_THAN", 0], "c2": ["LESS_THAN", 0.5]},
        )

        condition = FeasibilityCondition(require_all_constraints=True)

        # Test with empty data
        empty_data = pd.DataFrame()
        assert not condition.should_stop(empty_data, constrained_vocs)

        # Test with no constraints
        no_constraint_vocs = VOCS(
            variables={"x1": [0.0, 1.0], "x2": [0.0, 1.0]},
            objectives={"f1": "MINIMIZE"},
        )
        data = pd.DataFrame({"x1": [0.1], "x2": [0.2], "f1": [0.5]})
        assert not condition.should_stop(data, no_constraint_vocs)

        # Test with data that has feasible solutions
        feasible_data = pd.DataFrame(
            {
                "x1": [0.1, 0.2],
                "x2": [0.1, 0.2],
                "f1": [0.5, 0.4],
                "c1": [0.1, 0.2],  # satisfies constraint
                "c2": [0.2, 0.3],  # satisfies constraint
            }
        )

        assert condition.should_stop(feasible_data, constrained_vocs)

        # Test with require_all_constraints=False
        condition_partial = FeasibilityCondition(require_all_constraints=False)
        assert condition_partial.should_stop(feasible_data, constrained_vocs)

    def test_composite_condition_validators_and_serializers(self):
        """Test CompositeCondition validators and serializers."""
        # Test empty conditions list
        with pytest.raises(ValueError, match="At least one condition must be provided"):
            CompositeCondition(conditions=[], logic="or")

        # Test invalid logic
        with pytest.raises(ValueError, match="logic must be 'and' or 'or'"):
            CompositeCondition(
                conditions=[MaxEvaluationsCondition(max_evaluations=5)], logic="invalid"
            )

        # Test with dict conditions
        composite = CompositeCondition(
            conditions=[
                {"name": "MaxEvaluationsCondition", "max_evaluations": 10},
                MaxEvaluationsCondition(max_evaluations=5),
            ],
            logic="or",
        )
        assert len(composite.conditions) == 2

        # Test serialization
        serialized = composite.model_dump()
        assert "conditions" in serialized
        assert len(serialized["conditions"]) == 2
        assert serialized["conditions"][0]["name"] == "MaxEvaluationsCondition"

        # Test with invalid condition type
        with pytest.raises(
            ValueError,
            match="Each condition must be a StoppingCondition instance or a dict",
        ):
            CompositeCondition(conditions=["invalid"], logic="or")

    def test_composite_condition_logic_paths(self, simple_vocs):
        """Test CompositeCondition different logic paths."""
        data = pd.DataFrame(
            {
                "x1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "x2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "f1": [0.5, 0.4, 0.35, 0.34, 0.339, 0.338],
            }
        )

        # Test AND logic with both conditions true
        condition_and = CompositeCondition(
            conditions=[
                MaxEvaluationsCondition(max_evaluations=5),  # True
                TargetValueCondition(
                    objective_name="f1", target_value=0.5, tolerance=0.01
                ),  # True
            ],
            logic="and",
        )
        assert condition_and.should_stop(data, simple_vocs)

        # Test AND logic with one condition false
        condition_and_false = CompositeCondition(
            conditions=[
                MaxEvaluationsCondition(max_evaluations=10),  # False
                TargetValueCondition(
                    objective_name="f1", target_value=0.5, tolerance=0.01
                ),  # True
            ],
            logic="and",
        )
        assert not condition_and_false.should_stop(data, simple_vocs)

        # Test OR logic early return
        condition_or = CompositeCondition(
            conditions=[
                MaxEvaluationsCondition(
                    max_evaluations=5
                ),  # True - should return early
                TargetValueCondition(
                    objective_name="f1", target_value=0.1, tolerance=0.01
                ),  # False
            ],
            logic="or",
        )
        assert condition_or.should_stop(data, simple_vocs)

        # Test OR logic with all false
        condition_or_false = CompositeCondition(
            conditions=[
                MaxEvaluationsCondition(max_evaluations=10),  # False
                TargetValueCondition(
                    objective_name="f1", target_value=0.1, tolerance=0.01
                ),  # False
            ],
            logic="or",
        )
        assert not condition_or_false.should_stop(data, simple_vocs)

    def test_get_stopping_condition_function(self):
        """Test the get_stopping_condition utility function."""
        # Test valid condition names
        max_eval_class = get_stopping_condition("MaxEvaluationsCondition")
        assert max_eval_class == MaxEvaluationsCondition

        target_class = get_stopping_condition("TargetValueCondition")
        assert target_class == TargetValueCondition

        # Test invalid condition name
        with pytest.raises(
            ValueError, match="No stopping condition found with name: InvalidCondition"
        ):
            get_stopping_condition("InvalidCondition")

    def test_edge_cases_empty_and_missing_data(self, simple_vocs):
        """Test edge cases with empty data and missing columns."""
        # Test all conditions with empty dataframe
        empty_data = pd.DataFrame()

        conditions = [
            MaxEvaluationsCondition(max_evaluations=5),
            TargetValueCondition(objective_name="f1", target_value=0.1),
            ConvergenceCondition(
                objective_name="f1", improvement_threshold=0.01, patience=3
            ),
            StagnationCondition(objective_name="f1", patience=3, tolerance=0.01),
        ]

        for condition in conditions:
            if isinstance(condition, MaxEvaluationsCondition):
                assert not condition.should_stop(empty_data, simple_vocs)
            else:
                assert not condition.should_stop(empty_data, simple_vocs)

        # Test with missing objective column
        data_missing_obj = pd.DataFrame({"x1": [0.1], "x2": [0.2]})
        for condition in conditions[1:]:  # Skip MaxEvaluationsCondition
            assert not condition.should_stop(data_missing_obj, simple_vocs)

        # Test with wrong objective name in VOCS
        bad_vocs = VOCS(
            variables={"x1": [0.0, 1.0], "x2": [0.0, 1.0]},
            objectives={"wrong_obj": "MINIMIZE"},
        )
        data = pd.DataFrame({"x1": [0.1], "x2": [0.2], "f1": [0.5]})
        for condition in conditions[1:]:  # Skip MaxEvaluationsCondition
            assert not condition.should_stop(data, bad_vocs)
