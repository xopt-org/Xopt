from enum import Enum
from typing import Any, Dict, Union, List

import numpy as np
import pandas as pd
import yaml
from pydantic import conlist

from xopt.pydantic import XoptBaseModel


# Enums for objectives and constraints
class ObjectiveEnum(str, Enum):
    MINIMIZE = "MINIMIZE"
    MAXIMIZE = "MAXIMIZE"

    # Allow any case
    @classmethod
    def _missing_(cls, name):
        for member in cls:
            if member.name.lower() == name.lower():
                return member


class ConstraintEnum(str, Enum):
    LESS_THAN = "LESS_THAN"
    GREATER_THAN = "GREATER_THAN"

    # Allow any case
    @classmethod
    def _missing_(cls, name):
        for member in cls:
            if member.name.lower() == name.lower():
                return member


class VOCS(XoptBaseModel):
    """
    Variables, Objectives, Constraints, and other Settings (VOCS) data structure
    to describe optimization problems.
    """

    variables: Dict[str, conlist(float, min_items=2, max_items=2)] = {}
    constraints: Dict[
        str, conlist(Union[float, ConstraintEnum], min_items=2, max_items=2)
    ] = {}
    objectives: Dict[str, ObjectiveEnum] = {}
    constants: Dict[str, Any] = {}
    linked_variables: Dict[str, str] = {}

    class Config:
        validate_assignment = True  # Not sure this helps in this case
        use_enum_values = True

    @classmethod
    def from_yaml(cls, yaml_text):
        return cls.parse_obj(yaml.safe_load(yaml_text))

    def as_yaml(self):
        return yaml.dump(self.dict(), default_flow_style=None, sort_keys=False)

    @property
    def bounds(self):
        """
        Returns a bounds array (mins, maxs) of shape (2, n_variables)
        Arrays of lower and upper bounds can be extracted by:
            mins, maxs = vocs.bounds
        """
        return np.array([v for _, v in sorted(self.variables.items())]).T

    @property
    def variable_names(self):
        """Returns a sorted list of variable names"""
        return list(sorted(self.variables.keys()))

    @property
    def objective_names(self):
        """Returns a sorted list of objective names"""
        return list(sorted(self.objectives.keys()))

    @property
    def constraint_names(self):
        """Returns a sorted list of constraint names"""
        if self.constraints is None:
            return []
        return list(sorted(self.constraints.keys()))

    @property
    def output_names(self):
        """
        Returns a sorted list of objective and constraint names (objectives first
        then constraints)
        """
        return self.objective_names + self.constraint_names

    @property
    def constant_names(self):
        """Returns a sorted list of constraint names"""
        if self.constants is None:
            return []
        return list(sorted(self.constants.keys()))

    @property
    def all_names(self):
        """Returns all vocs names (variables, constants, objectives, constraints"""
        return (
            self.variable_names
            + self.constant_names
            + self.objective_names
            + self.constraint_names
        )

    @property
    def n_variables(self):
        """Returns the number of variables"""
        return len(self.variables)

    @property
    def n_constants(self):
        """Returns the number of constants"""
        return len(self.constants)

    @property
    def n_inputs(self):
        """Returns the number of inputs (variables and constants)"""
        return self.n_variables + self.n_constants

    @property
    def n_objectives(self):
        """Returns the number of objectives"""
        return len(self.objectives)

    @property
    def n_constraints(self):
        """Returns the number of constraints"""
        return len(self.constraints)

    @property
    def n_outputs(self):
        """Returns the number of outputs (objectives and constraints)"""
        return self.n_objectives + self.n_constraints

    def random_inputs(
        self, n=None, include_constants=True, include_linked_variables=True
    ):
        """
        Uniform sampling of the variables.

        Returns a dict of inputs.

        If include_constants, the vocs.constants are added to the dict.

        Optional:
            n (integer) to make arrays of inputs, of size n.

        """
        inputs = {}
        for key, val in self.variables.items():  # No need to sort here
            a, b = val
            x = np.random.random(n)
            inputs[key] = x * a + (1 - x) * b

        # Constants
        if include_constants and self.constants is not None:
            inputs.update(self.constants)

        # Handle linked variables
        if include_linked_variables and self.linked_variables is not None:
            for k, v in self.linked_variables.items():
                inputs[k] = inputs[v]

        # return pd.DataFrame(inputs, index=range(n))
        return inputs

    def convert_dataframe_to_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts only inputs from a dataframe.
        This will add constants.
        """
        # make sure that the df keys contain the vocs variables
        if not set(self.variable_names).issubset(set(data.keys())):
            raise RuntimeError(
                "input dataframe must at least contain the vocs variables"
            )

        # only keep the variables
        in_copy = data[self.variable_names].copy()

        # append constants
        constants = self.constants
        if constants is not None:
            for name, val in constants.items():
                in_copy[name] = val

        return in_copy

    def convert_numpy_to_inputs(self, inputs: np.ndarray) -> pd.DataFrame:
        """
        convert 2D numpy array to list of dicts (inputs) for evaluation
        Assumes that the columns of the array match correspond to
        `sorted(self.vocs.variables.keys())

        """
        df = pd.DataFrame(inputs, columns=self.variable_names)
        return self.convert_dataframe_to_inputs(df)

    # Extract optimization data (in correct column order)
    def variable_data(
        self,
        data: Union[pd.DataFrame, List[Dict], List[Dict]],
        prefix: str = "variable_",
    ) -> pd.DataFrame:
        """
        Returns a dataframe containing variables according to `vocs.variables` in sorted
        order

        Args:
            data: Data to be processed.
            prefix: Prefix added to column names.

        Returns:
            result: processed Dataframe
        """
        return form_variable_data(self.variables, data, prefix=prefix)

    def objective_data(
        self,
        data: Union[pd.DataFrame, List[Dict], List[Dict]],
        prefix: str = "objective_",
    ) -> pd.DataFrame:
        """
        Returns a dataframe containing objective data transformed according to
        `vocs.objectives` such that we always assume minimization.

        Args:
            data: data to be processed.
            prefix: prefix added to column names.

        Returns:
            result: processed Dataframe
        """
        return form_objective_data(self.objectives, data, prefix)

    def constraint_data(
        self,
        data: Union[pd.DataFrame, List[Dict], List[Dict]],
        prefix: str = "constraint_",
    ) -> pd.DataFrame:
        """
        Returns a dataframe containing constraint data transformed according to
        `vocs.constraints` such that values that satisfy each constraint are negative.

        Args:
            data: data to be processed.
            prefix: prefix added to column names.

        Returns:
            result: processed Dataframe
        """
        return form_constraint_data(self.constraints, data, prefix)

    def feasibility_data(
        self,
        data: Union[pd.DataFrame, List[Dict], List[Dict]],
        prefix: str = "feasible_",
    ) -> pd.DataFrame:
        """
        Returns a dataframe containing booleans denoting if a constraint is satisfied or
        not. Returned dataframe also contains a column `feasibility` which denotes if
        all constraints are satisfied.

        Args:
            data: data to be processed.
            prefix: prefix added to column names.

        Returns:
            result: processed Dataframe
        """
        return form_feasibility_data(self.constraints, data, prefix)


# --------------------------------
# dataframe utilities

OBJECTIVE_WEIGHT = {"MINIMIZE": 1.0, "MAXIMIZE": -1.0}


def form_variable_data(variables: Dict, data, prefix="variable_"):
    """
    Use variables dict to form a dataframe.
    """
    if not variables:
        return None

    data = pd.DataFrame(data)
    vdata = pd.DataFrame()
    for k in sorted(list(variables)):
        vdata[prefix + k] = data[k]

    return vdata


def form_objective_data(objectives: Dict, data, prefix="objective_"):
    """
    Use objective dict and data (dataframe) to generate objective data (dataframe)

    Weights are applied to convert all objectives into mimimization form.

    Returns a dataframe with the objective data intented to be minimized.

    Missing or nan values will be filled with: np.inf

    """
    if not objectives:
        return None

    data = pd.DataFrame(data)

    odata = pd.DataFrame(index=data.index)

    for k in sorted(list(objectives)):

        # Protect against missing data
        if k not in data:
            odata[prefix + k] = np.inf
            continue

        operator = objectives[k].upper()
        if operator not in OBJECTIVE_WEIGHT:
            raise ValueError(f"Unknown objective operator: {operator}")

        weight = OBJECTIVE_WEIGHT[operator]
        odata[prefix + k] = (weight * data[k]).fillna(np.inf)  # Protect against nans

    return odata


def form_constraint_data(constraints: Dict, data: pd.DataFrame, prefix="constraint_"):
    """
    Use constraint dict and data (dataframe) to generate constraint data (dataframe). A
    constraint is satisfied if the evaluation is < 0.

    Args:
        constraints: Dictonary of constraints
        data: Dataframe with the data to be evaluated
        prefix: Prefix to use for the transformed data in the dataframe

    Returns a dataframe with the constraint data.

    Missing or nan values will be filled with: np.inf

    """
    if not constraints:
        return None

    data = pd.DataFrame(data)  # cast to dataframe

    cdata = pd.DataFrame(index=data.index)

    for k in sorted(list(constraints)):

        # Protect against missing data
        if k not in data:
            cdata[prefix + k] = np.inf
            continue

        x = data[k]
        op, d = constraints[k]
        op = op.upper()  # Allow any case

        if op == "GREATER_THAN":  # x > d -> x-d > 0
            cvalues = -(x - d)
        elif op == "LESS_THAN":  # x < d -> d-x > 0
            cvalues = -(d - x)
        else:
            raise ValueError(f"Unknown constraint operator: {op}")

        cdata[prefix + k] = cvalues.fillna(np.inf)  # Protect against nans
    return cdata


def form_feasibility_data(constraints: Dict, data, prefix="feasible_"):
    """
    Use constraint dict and data to identify feasible points in the the dataset.

    Returns a dataframe with the feasibility data.
    """
    if not constraints:
        df = pd.DataFrame(index=data.index)
        df["feasible"] = True
        return df

    data = pd.DataFrame(data)
    c_prefix = "constraint_"
    cdata = form_constraint_data(constraints, data, prefix=c_prefix)
    fdata = pd.DataFrame()
    for k in sorted(list(constraints)):
        fdata[prefix + k] = cdata[c_prefix + k] <= 0
    # if all row values are true, then the row is feasible
    fdata["feasible"] = fdata.all(axis=1)
    return fdata
