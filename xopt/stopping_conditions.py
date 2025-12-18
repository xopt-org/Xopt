"""
Stopping conditions for Xopt optimization.

This module contains classes that implement various stopping criteria for optimization
processes. Each stopping condition class takes an Xopt dataframe and VOCS object to
determine whether optimization should stop.
"""

from abc import ABC, abstractmethod
from typing import List

import pandas as pd
from pydantic import (
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    field_serializer,
    field_validator,
)


from xopt.pydantic import XoptBaseModel
from xopt.vocs import get_feasibility_data
from gest_api.vocs import MinimizeObjective, VOCS


class StoppingCondition(XoptBaseModel, ABC):
    """
    Abstract base class for stopping conditions.

    All stopping conditions must implement the should_stop method which takes
    an Xopt dataframe and VOCS object and returns True if optimization should stop.
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    @abstractmethod
    def should_stop(self, data: pd.DataFrame, vocs: VOCS) -> bool:
        """
        Determine if optimization should stop based on data and VOCS.

        Parameters
        ----------
        data : pd.DataFrame
            The Xopt optimization data containing variables, objectives, constraints, etc.
        vocs : VOCS
            The VOCS object defining variables, objectives, and constraints.

        Returns
        -------
        bool
            True if optimization should stop, False otherwise.
        """
        ...


def get_stopping_condition(name: str) -> type[StoppingCondition]:
    """
    Retrieve a stopping condition class by its name.

    Parameters
    ----------
    name : str
        Name of the stopping condition class.

    Returns
    -------
    type[StoppingCondition]
        The stopping condition class.

    Raises
    ------
    ValueError
        If no stopping condition with the given name is found.
    """
    subclasses = StoppingCondition.__subclasses__()
    for subclass in subclasses:
        if subclass.__name__ == name:
            return subclass
    raise ValueError(f"No stopping condition found with name: {name}")


class MaxEvaluationsCondition(StoppingCondition):
    """
    Stop after a maximum number of evaluations. Evaluations can be counted
    in different ways based on parameters. If count_valid_only is True, only
    evaluations without errors are counted. If use_dataframe_index is True,
    the dataframe index is used to count evaluations instead of the number of rows.

    Parameters
    ----------
    max_evaluations : int
        Maximum number of function evaluations before stopping.
    count_valid_only : bool, optional
        Whether to count only valid evaluations (default: False).
    use_dataframe_index : bool, optional
        Whether to use the dataframe index to count evaluations (default: False).
    """

    max_evaluations: PositiveInt = Field(
        description="Maximum number of function evaluations"
    )
    count_valid_only: bool = Field(
        default=False,
        description="Whether to count only valid evaluations that do not raise errors",
    )
    use_dataframe_index: bool = Field(
        default=False,
        description="Whether to use the dataframe index to count evaluations",
    )

    def should_stop(self, data: pd.DataFrame, vocs: VOCS) -> bool:
        """Stop if we've reached the maximum number of evaluations."""
        if data.empty:
            return False

        if self.use_dataframe_index:
            # assume index starts at 0
            return max(data.index) + 1 >= self.max_evaluations

        if self.count_valid_only:
            valid_data = data[data["error"].isna()] if "error" in data.columns else data
            return len(valid_data) >= self.max_evaluations

        return len(data) >= self.max_evaluations


class TargetValueCondition(StoppingCondition):
    """
    Stop when an objective reaches a target value.

    Parameters
    ----------
    objective_name : str
        Name of the objective to monitor.
    target_value : float
        Target value for the objective.
    tolerance : float, optional
        Tolerance for reaching the target (default: 1e-6).
    """

    objective_name: str = Field(description="Name of the objective to monitor")
    target_value: float = Field(description="Target value for the objective")
    tolerance: PositiveFloat = Field(
        default=1e-6, description="Tolerance for reaching the target"
    )

    def should_stop(self, data: pd.DataFrame, vocs: VOCS) -> bool:
        """Stop if objective reaches target value within tolerance."""
        if data.empty or self.objective_name not in vocs.objectives:
            return False

        if self.objective_name not in data.columns:
            return False

        objective_type = vocs.objectives[self.objective_name]
        objective_values = data[self.objective_name].dropna()

        if len(objective_values) == 0:
            return False

        if isinstance(objective_type, MinimizeObjective):
            best_value = objective_values.min()
            return best_value <= self.target_value + self.tolerance
        else:  # MAXIMIZE
            best_value = objective_values.max()
            return best_value >= self.target_value - self.tolerance


class ConvergenceCondition(StoppingCondition):
    """
    Stop when optimization converges (improvement is below threshold).

    Parameters
    ----------
    objective_name : str
        Name of the objective to monitor for convergence.
    improvement_threshold : float
        Minimum improvement required to continue optimization.
    patience : int
        Number of evaluations to wait without improvement before stopping.
    relative : bool, optional
        Whether to use relative improvement (default: False).
    """

    objective_name: str = Field(description="Name of the objective to monitor")
    improvement_threshold: PositiveFloat = Field(
        description="Minimum improvement required to continue optimization"
    )
    patience: PositiveInt = Field(
        description="Number of evaluations to wait without improvement"
    )
    relative: bool = Field(
        default=False, description="Whether to use relative improvement"
    )

    def should_stop(self, data: pd.DataFrame, vocs: VOCS) -> bool:
        """Stop if no improvement for specified patience."""
        if data.empty or self.objective_name not in vocs.objectives:
            return False

        if self.objective_name not in data.columns:
            return False

        if len(data) < self.patience + 1:
            return False

        objective_type = vocs.objectives[self.objective_name]
        objective_values = data[self.objective_name].dropna()

        # Check improvement over the last 'patience' evaluations
        recent_values = objective_values.iloc[-(self.patience + 1) :]

        if isinstance(objective_type, MinimizeObjective):
            best_recent = recent_values.min()
            baseline = recent_values.iloc[0]
            improvement = baseline - best_recent
        else:  # MAXIMIZE
            best_recent = recent_values.max()
            baseline = recent_values.iloc[0]
            improvement = best_recent - baseline

        if self.relative and abs(baseline) > 1e-12:
            improvement = improvement / abs(baseline)

        return improvement < self.improvement_threshold


class StagnationCondition(StoppingCondition):
    """
    Stop when the best objective value hasn't improved for a number of evaluations.

    Parameters
    ----------
    objective_name : str
        Name of the objective to monitor.
    patience : int
        Number of evaluations without improvement before stopping.
    tolerance : float, optional
        Minimum improvement considered significant (default: 1e-8).
    """

    objective_name: str = Field(description="Name of the objective to monitor")
    patience: PositiveInt = Field(
        description="Number of evaluations without improvement before stopping"
    )
    tolerance: PositiveFloat = Field(
        default=1e-8, description="Minimum improvement considered significant"
    )

    def should_stop(self, data: pd.DataFrame, vocs: VOCS) -> bool:
        """Stop if no improvement in best value for specified patience."""
        if data.empty or self.objective_name not in vocs.objectives:
            return False

        if self.objective_name not in data.columns:
            return False

        if len(data) < self.patience + 1:
            return False

        objective_type = vocs.objectives[self.objective_name]
        objective_values = data[self.objective_name].dropna()

        # Track the best value seen so far
        if isinstance(objective_type, MinimizeObjective):
            cumulative_best = objective_values.cummin()
        else:  # MAXIMIZE
            cumulative_best = objective_values.cummax()

        recent_best = cumulative_best.iloc[-1]
        past_best = cumulative_best.iloc[-(self.patience + 1)]

        if isinstance(objective_type, MinimizeObjective):
            improvement = past_best - recent_best
        else:  # MAXIMIZE
            improvement = recent_best - past_best

        return improvement < self.tolerance


class FeasibilityCondition(StoppingCondition):
    """
    Stop when a feasible solution is found.

    Parameters
    ----------
    require_all_constraints : bool, optional
        Whether all constraints must be satisfied (default: True).
    """

    require_all_constraints: bool = Field(
        default=True, description="Whether all constraints must be satisfied"
    )

    def should_stop(self, data: pd.DataFrame, vocs: VOCS) -> bool:
        """Stop when a feasible solution is found."""
        if data.empty or not vocs.constraints:
            return False

        # Use VOCS to determine feasibility
        feasibility_data = get_feasibility_data(vocs, data)

        if self.require_all_constraints:
            # Stop if any point is fully feasible
            return feasibility_data["feasible"].any()
        else:
            # Stop if any individual constraint is satisfied
            constraint_columns = [
                col
                for col in feasibility_data.columns
                if col.startswith("feasible_") and col != "feasible"
            ]
            if constraint_columns:
                return feasibility_data[constraint_columns].any().any()


class CompositeCondition(StoppingCondition):
    """
    Combine multiple stopping conditions with AND/OR logic.

    Parameters
    ----------
    conditions : List[StoppingCondition]
        List of stopping conditions to combine.
    logic : str, optional
        Logic to combine conditions: "and" or "or" (default: "or").
    """

    conditions: List[StoppingCondition] = Field(
        description="List of stopping conditions to combine"
    )
    logic: str = Field(
        default="or", description="Logic to combine conditions: 'and' or 'or'"
    )

    @field_validator("conditions", mode="before")
    def validate_conditions(cls, v):
        if len(v) == 0:
            raise ValueError("At least one condition must be provided")

        # process list to ensure all are StoppingCondition instances
        validated_conditions = []
        for condition in v:
            if isinstance(condition, dict):
                name = condition.pop("name")
                sc_class = get_stopping_condition(name)
                condition = sc_class(**condition)
            elif isinstance(condition, StoppingCondition):
                pass
            else:
                raise ValueError(
                    "Each condition must be a StoppingCondition instance or a dict"
                )
            validated_conditions.append(condition)

        return validated_conditions

    @field_serializer("conditions")
    @classmethod
    def serialize_conditions(cls, v):
        serialized_conditions = []
        for condition in v:
            serialized_conditions.append(
                condition.model_dump() | {"name": condition.__class__.__name__}
            )
        return serialized_conditions

    @field_validator("logic")
    @classmethod
    def validate_logic(cls, v):
        if v.lower() not in ["and", "or"]:
            raise ValueError("logic must be 'and' or 'or'")
        return v.lower()

    def should_stop(self, data: pd.DataFrame, vocs: VOCS) -> bool:
        """Combine conditions using specified logic."""
        results = []

        for condition in self.conditions:
            _should_stop = condition.should_stop(data, vocs)
            if _should_stop and self.logic == "or":
                return True
            results.append(_should_stop)

        if self.logic == "and":
            return all(results)
        return False
