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
from xopt.stopping_conditions import (
    CompositeCondition,
    ConvergenceCondition,
    MaxEvaluationsCondition,
    StagnationCondition,
    TargetValueCondition,
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

        condition = ConvergenceCondition(
            objective_name="f1", improvement_threshold=0.001, patience=2
        )
        assert not condition.should_stop(data, simple_vocs)

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
            MaxEvaluationsCondition(max_evaluations=10),
            TargetValueCondition(objective_name="f1", target_value=0.1, tolerance=0.01),
            ConvergenceCondition(
                objective_name="f1", improvement_threshold=0.01, patience=5
            ),
            StagnationCondition(objective_name="f1", patience=5, tolerance=1e-8),
            CompositeCondition(
                conditions=[
                    MaxEvaluationsCondition(max_evaluations=10),
                    TargetValueCondition(
                        objective_name="f1", target_value=0.1, tolerance=0.01
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
        generator = LatinHypercubeGenerator(vocs=simple_vocs)

        X = Xopt(
            vocs=simple_vocs,
            evaluator=evaluator,
            generator=generator,
            stopping_condition=condition,
        )

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
