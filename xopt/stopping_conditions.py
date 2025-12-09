"""
Stopping conditions for Xopt optimization.

This module contains classes that implement various stopping criteria for optimization
processes. Each stopping condition class takes an Xopt dataframe and VOCS object to
determine whether optimization should stop.
"""

from abc import ABC, abstractmethod
from typing import Annotated, List

import pandas as pd
from pydantic import (
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    SerializeAsAny,
    field_serializer,
    field_validator,
)

from xopt.pydantic import XoptBaseModel
from xopt.vocs import VOCS


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
        pass


# Type alias for lists of stopping conditions with proper serialization
StoppingConditionList = List[Annotated[StoppingCondition, SerializeAsAny]]


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
    Stop after a maximum number of evaluations.

    Parameters
    ----------
    max_evaluations : int
        Maximum number of function evaluations before stopping.
    """

    max_evaluations: PositiveInt = Field(
        description="Maximum number of function evaluations"
    )

    def should_stop(self, data: pd.DataFrame, vocs: VOCS) -> bool:
        """Stop if we've reached the maximum number of evaluations."""
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

        if objective_type.upper() == "MINIMIZE":
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

    objective_name: str = Field(
        description="Name of the objective to monitor for convergence"
    )
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

        if len(objective_values) < self.patience + 1:
            return False

        # Check improvement over the last 'patience' evaluations
        recent_values = objective_values.iloc[-(self.patience + 1) :]

        if objective_type.upper() == "MINIMIZE":
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
    patience: int = Field(
        description="Number of evaluations without improvement before stopping"
    )
    tolerance: float = Field(
        default=1e-8, description="Minimum improvement considered significant"
    )

    @field_validator("patience")
    @classmethod
    def validate_patience(cls, v):
        if v <= 0:
            raise ValueError("patience must be positive")
        return v

    @field_validator("tolerance")
    @classmethod
    def validate_tolerance(cls, v):
        if v < 0:
            raise ValueError("tolerance must be non-negative")
        return v

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

        if len(objective_values) < self.patience + 1:
            return False

        # Track the best value seen so far
        if objective_type.upper() == "MINIMIZE":
            cumulative_best = objective_values.cummin()
        else:  # MAXIMIZE
            cumulative_best = objective_values.cummax()

        # Check if there's been improvement in the last 'patience' evaluations
        if len(cumulative_best) < self.patience:
            return False

        recent_best = cumulative_best.iloc[-1]
        past_best = cumulative_best.iloc[-(self.patience + 1)]

        if objective_type.upper() == "MINIMIZE":
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
        feasibility_data = vocs.feasibility_data(data)

        if "feasible" not in feasibility_data.columns:
            return False

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

        return False


class ObjectiveThresholdCondition(StoppingCondition):
    """
    Stop when an objective crosses a threshold value.

    Parameters
    ----------
    objective_name : str
        Name of the objective to monitor.
    threshold : float
        Threshold value for the objective.
    direction : str, optional
        Direction of threshold crossing: "below", "above", or "either" (default: "either").
    """

    objective_name: str = Field(description="Name of the objective to monitor")
    threshold: float = Field(description="Threshold value for the objective")
    direction: str = Field(
        default="either",
        description="Direction of threshold crossing: 'below', 'above', or 'either'",
    )

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v):
        if v.lower() not in ["below", "above", "either"]:
            raise ValueError("direction must be 'below', 'above', or 'either'")
        return v.lower()

    def should_stop(self, data: pd.DataFrame, vocs: VOCS) -> bool:
        """Stop when objective crosses threshold."""
        if data.empty or self.objective_name not in vocs.objectives:
            return False

        if self.objective_name not in data.columns:
            return False

        objective_values = data[self.objective_name].dropna()

        if len(objective_values) == 0:
            return False

        if self.direction == "below":
            return (objective_values < self.threshold).any()
        elif self.direction == "above":
            return (objective_values > self.threshold).any()
        else:  # "either"
            return (
                (objective_values < self.threshold)
                | (objective_values > self.threshold)
            ).any()


class CompositeCondition(StoppingCondition):
    """
    Combine multiple stopping conditions with AND/OR logic.

    Parameters
    ----------
    conditions : StoppingConditionList
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
        results = [condition.should_stop(data, vocs) for condition in self.conditions]

        if self.logic == "and":
            return all(results)
        else:  # "or"
            return any(results)
