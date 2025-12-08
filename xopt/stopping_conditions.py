"""
Stopping conditions for Xopt optimization.

This module contains classes that implement various stopping criteria for optimization
processes. Each stopping condition class takes an Xopt dataframe and VOCS object to
determine whether optimization should stop.
"""

from abc import ABC, abstractmethod
from typing import List
import pandas as pd
from pydantic import Field, field_validator, ConfigDict

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


class MaxEvaluationsCondition(StoppingCondition):
    """
    Stop after a maximum number of evaluations.

    Parameters
    ----------
    max_evaluations : int
        Maximum number of function evaluations before stopping.
    """

    max_evaluations: int = Field(description="Maximum number of function evaluations")

    @field_validator("max_evaluations")
    @classmethod
    def validate_max_evaluations(cls, v):
        if v <= 0:
            raise ValueError("max_evaluations must be positive")
        return v

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
    tolerance: float = Field(
        default=1e-6, description="Tolerance for reaching the target"
    )

    @field_validator("tolerance")
    @classmethod
    def validate_tolerance(cls, v):
        if v < 0:
            raise ValueError("tolerance must be non-negative")
        return v

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
    improvement_threshold: float = Field(
        description="Minimum improvement required to continue optimization"
    )
    patience: int = Field(
        description="Number of evaluations to wait without improvement"
    )
    relative: bool = Field(
        default=False, description="Whether to use relative improvement"
    )

    @field_validator("improvement_threshold")
    @classmethod
    def validate_improvement_threshold(cls, v):
        if v < 0:
            raise ValueError("improvement_threshold must be non-negative")
        return v

    @field_validator("patience")
    @classmethod
    def validate_patience(cls, v):
        if v <= 0:
            raise ValueError("patience must be positive")
        return v

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

        if len(data) < self.patience:
            return False

        objective_type = vocs.objectives[self.objective_name]
        objective_values = data[self.objective_name].dropna()

        if len(objective_values) < self.patience:
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


class VarianceCondition(StoppingCondition):
    """
    Stop when the variance of recent objective values falls below threshold.

    Parameters
    ----------
    objective_name : str
        Name of the objective to monitor.
    variance_threshold : float
        Minimum variance threshold.
    window_size : int
        Number of recent evaluations to consider for variance calculation.
    """

    objective_name: str = Field(description="Name of the objective to monitor")
    variance_threshold: float = Field(description="Minimum variance threshold")
    window_size: int = Field(description="Number of recent evaluations to consider")

    @field_validator("variance_threshold")
    @classmethod
    def validate_variance_threshold(cls, v):
        if v < 0:
            raise ValueError("variance_threshold must be non-negative")
        return v

    @field_validator("window_size")
    @classmethod
    def validate_window_size(cls, v):
        if v <= 1:
            raise ValueError("window_size must be greater than 1")
        return v

    def should_stop(self, data: pd.DataFrame, vocs: VOCS) -> bool:
        """Stop when variance of recent values is below threshold."""
        if data.empty or self.objective_name not in vocs.objectives:
            return False

        if self.objective_name not in data.columns:
            return False

        objective_values = data[self.objective_name].dropna()

        if len(objective_values) < self.window_size:
            return False

        # Calculate variance of the most recent window_size values
        recent_values = objective_values.iloc[-self.window_size :]
        variance = recent_values.var()

        return variance < self.variance_threshold


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

    @field_validator("conditions")
    @classmethod
    def validate_conditions(cls, v):
        if len(v) == 0:
            raise ValueError("At least one condition must be provided")
        return v

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


class RelativeImprovementCondition(StoppingCondition):
    """
    Stop when relative improvement falls below threshold.

    Parameters
    ----------
    objective_name : str
        Name of the objective to monitor.
    relative_threshold : float
        Relative improvement threshold (e.g., 0.01 for 1% improvement).
    window_size : int
        Number of evaluations to compare for improvement.
    """

    objective_name: str = Field(description="Name of the objective to monitor")
    relative_threshold: float = Field(description="Relative improvement threshold")
    window_size: int = Field(description="Number of evaluations to compare")

    @field_validator("relative_threshold")
    @classmethod
    def validate_relative_threshold(cls, v):
        if v < 0:
            raise ValueError("relative_threshold must be non-negative")
        return v

    @field_validator("window_size")
    @classmethod
    def validate_window_size(cls, v):
        if v <= 1:
            raise ValueError("window_size must be greater than 1")
        return v

    def should_stop(self, data: pd.DataFrame, vocs: VOCS) -> bool:
        """Stop when relative improvement is below threshold."""
        if data.empty or self.objective_name not in vocs.objectives:
            return False

        if self.objective_name not in data.columns:
            return False

        if len(data) < self.window_size:
            return False

        objective_type = vocs.objectives[self.objective_name]
        objective_values = data[self.objective_name].dropna()

        if len(objective_values) < self.window_size:
            return False

        # Compare best value in recent window with baseline
        recent_values = objective_values.iloc[-self.window_size :]
        baseline = objective_values.iloc[-(self.window_size + 1)]

        if objective_type.upper() == "MINIMIZE":
            best_recent = recent_values.min()
            improvement = baseline - best_recent
        else:  # MAXIMIZE
            best_recent = recent_values.max()
            improvement = best_recent - baseline

        if abs(baseline) > 1e-12:
            relative_improvement = improvement / abs(baseline)
        else:
            relative_improvement = 0.0

        return relative_improvement < self.relative_threshold
