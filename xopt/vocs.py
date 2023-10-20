from enum import Enum
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import yaml
from pydantic import ConfigDict, conlist, Field

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

    variables: Dict[str, conlist(float, min_length=2, max_length=2)] = Field(
        default={},
        description="input variable names with a list of minimum and maximum values",
    )
    constraints: Dict[
        str, conlist(Union[float, ConstraintEnum], min_length=2, max_length=2)
    ] = Field(
        default={},
        description="constraint names with a list of constraint type and value",
    )
    objectives: Dict[str, ObjectiveEnum] = Field(
        default={}, description="objective names with type of objective"
    )
    constants: Dict[str, Any] = Field(
        default={}, description="constant names and values passed to evaluate function"
    )
    observables: List[str] = Field(
        default=[],
        description="observation names tracked alongside objectives and constraints",
    )

    model_config = ConfigDict(
        validate_assignment=True, use_enum_values=True, extra="forbid"
    )

    @classmethod
    def from_yaml(cls, yaml_text):
        loaded = yaml.safe_load(yaml_text)
        return cls(**loaded)

    def as_yaml(self):
        return yaml.dump(self.model_dump(), default_flow_style=None, sort_keys=False)

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
    def observable_names(self):
        return sorted(self.observables)

    @property
    def output_names(self):
        """
        Returns a list of expected output keys:
            (objectives + constraints + observables)
        Each sub-list is sorted.
        """
        full_list = self.objective_names
        for ele in self.constraint_names:
            if ele not in full_list:
                full_list += [ele]

        for ele in self.observable_names:
            if ele not in full_list:
                full_list += [ele]

        return full_list

    @property
    def constant_names(self):
        """Returns a sorted list of constraint names"""
        if self.constants is None:
            return []
        return list(sorted(self.constants.keys()))

    @property
    def all_names(self):
        """Returns all vocs names (variables, constants, objectives, constraints)"""
        return self.variable_names + self.constant_names + self.output_names

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
    def n_observables(self):
        """Returns the number of constraints"""
        return len(self.observables)

    @property
    def n_outputs(self):
        """
        Returns the number of outputs
            len(objectives + constraints + observables)
        """
        return len(self.output_names)

    def random_inputs(
        self,
        n: int = None,
        custom_bounds: dict = None,
        include_constants: bool = True,
        seed: int = None,
    ) -> list[dict]:
        """
        Uniform sampling of the variables.

        Returns a dict of inputs.

        If include_constants, the vocs.constants are added to the dict.

        Optional:
            n (integer) to make arrays of inputs, of size n.
            seed (integer) to initialize the random number generator

        """
        inputs = {}
        if seed is None:
            rng_sample_function = np.random.random
        else:
            rng = np.random.default_rng(seed=seed)
            rng_sample_function = rng.random

        # get bounds
        # if custom_bounds is specified then they will be clipped inside
        # vocs variable bounds
        if custom_bounds is None:
            bounds = self.variables
        else:
            variable_bounds = pd.DataFrame(self.variables)
            custom_bounds = pd.DataFrame(custom_bounds)
            custom_bounds = custom_bounds.clip(
                variable_bounds.iloc[0], variable_bounds.iloc[1], axis=1
            )
            bounds = custom_bounds.to_dict()
            for k in bounds.keys():
                bounds[k] = [bounds[k][i] for i in range(2)]

        for key, val in bounds.items():  # No need to sort here
            a, b = val
            n = n if n is not None else 1
            x = rng_sample_function(n)
            inputs[key] = x * a + (1 - x) * b

        # Constants
        if include_constants and self.constants is not None:
            inputs.update(self.constants)

        if n == 1:
            return [inputs]
        else:
            return pd.DataFrame(inputs).to_dict("records")

    def convert_dataframe_to_inputs(
        self, data: pd.DataFrame, include_constants=True
    ) -> pd.DataFrame:
        """
        Extracts only inputs from a dataframe.
        This will add constants if `include_constants` is true.
        """
        # make sure that the df keys only contain vocs variables
        if not set(self.variable_names) == set(data.keys()):
            raise ValueError(
                "input dataframe column set must equal set of vocs variables"
            )

        # only keep the variables
        inner_copy = data.copy()

        # append constants if requested
        if include_constants:
            constants = self.constants
            if constants is not None:
                for name, val in constants.items():
                    inner_copy[name] = val

        return inner_copy

    def convert_numpy_to_inputs(
        self, inputs: np.ndarray, include_constants=True
    ) -> pd.DataFrame:
        """
        convert 2D numpy array to list of dicts (inputs) for evaluation
        Assumes that the columns of the array match correspond to
        `sorted(self.vocs.variables.keys())

        """
        df = pd.DataFrame(inputs, columns=self.variable_names)
        return self.convert_dataframe_to_inputs(df, include_constants)

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
        return_raw=False,
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
        return form_objective_data(self.objectives, data, prefix, return_raw)

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

    def observable_data(
        self,
        data: Union[pd.DataFrame, List[Dict], List[Dict]],
        prefix: str = "observable_",
    ) -> pd.DataFrame:
        """
        Returns a dataframe containing observable data

        Args:
            data: data to be processed.
            prefix: prefix added to column names.

        Returns:
            result: processed Dataframe
        """
        return form_observable_data(self.observable_names, data, prefix)

    def feasibility_data(
        self,
        data: Union[pd.DataFrame, List[Dict], List[Dict]],
        prefix: str = "feasible_",
    ) -> pd.DataFrame:
        """
        Returns a dataframe containing booleans denoting if a constraint is satisfied or
        not. Returned dataframe also contains a column `feasible` which denotes if
        all constraints are satisfied.

        Args:
            data: data to be processed.
            prefix: prefix added to column names.

        Returns:
            result: processed Dataframe
        """
        return form_feasibility_data(self.constraints, data, prefix)

    def validate_input_data(self, input_points: pd.DataFrame) -> None:
        """
        Validates input data. Raises an error if the input data does not satisfy
        requirements given by vocs.

        Args:
            input_points: input data to be validated.

        Returns:
            None

        Raises:
            ValueError: if input data does not satisfy requirements.
        """
        validate_input_data(self, input_points)

    def extract_data(self, data: pd.DataFrame, return_raw=False):
        """
        split dataframe into seperate dataframes for variables, objectives and
        constraints based on vocs - objective data is transformed based on
        `vocs.objectives` properties

        Args:
            data: dataframe to be split
            return_raw: if True, return untransformed objective data

        Returns:
            variable_data: dataframe containing variable data
            objective_data: dataframe containing objective data
            constraint_data: dataframe containing constraint data
        """
        variable_data = self.variable_data(data, "")
        objective_data = self.objective_data(data, "", return_raw)
        constraint_data = self.constraint_data(data, "")
        return variable_data, objective_data, constraint_data

    def select_best(self, data: pd.DataFrame, n=1):
        """
        get the best value and point for a given data set based on vocs
        - does not work for multi-objective problems
        - data that violates any constraints is ignored

        Args:
            data: dataframe to select best point from
            n: number of best points to return

        Returns:
            index: index of best point
            value: value of best point
        """
        if self.n_objectives != 1:
            raise NotImplementedError(
                "cannot select best point when n_objectives is not 1"
            )

        feasible_data = self.feasibility_data(data)
        ascending_flag = {"MINIMIZE": True, "MAXIMIZE": False}
        obj = self.objectives[self.objective_names[0]]
        obj_name = self.objective_names[0]
        res = data[feasible_data["feasible"]].sort_values(
            obj_name, ascending=ascending_flag[obj]
        )[obj_name][:n]

        return res.index.to_numpy(), res.to_numpy()


# --------------------------------
# dataframe utilities

OBJECTIVE_WEIGHT = {"MINIMIZE": 1.0, "MAXIMIZE": -1.0}


def form_variable_data(variables: Dict, data, prefix="variable_"):
    """
    Use variables dict to form a dataframe.
    """
    if not variables:
        return pd.DataFrame([])

    data = pd.DataFrame(data)
    vdata = pd.DataFrame()
    for k in sorted(list(variables)):
        vdata[prefix + k] = data[k]

    return vdata


def form_objective_data(
    objectives: Dict, data, prefix="objective_", return_raw: bool = False
):
    """
    Use objective dict and data (dataframe) to generate objective data (dataframe)

    Weights are applied to convert all objectives into mimimization form unless
    `return_raw` is True

    Returns a dataframe with the objective data intented to be minimized.

    Missing or nan values will be filled with: np.inf

    """
    if not objectives:
        return pd.DataFrame([])

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

        weight = 1.0 if return_raw else OBJECTIVE_WEIGHT[operator]
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
        return pd.DataFrame([])

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


def form_observable_data(observables: List, data: pd.DataFrame, prefix="observable_"):
    """
    Use constraint dict and data (dataframe) to generate constraint data (dataframe). A
    constraint is satisfied if the evaluation is < 0.

    Args:
        observables: Dictonary of constraints
        data: Dataframe with the data to be evaluated
        prefix: Prefix to use for the transformed data in the dataframe

    Returns a dataframe with the constraint data.

    Missing or nan values will be filled with: np.inf

    """
    if not observables:
        return pd.DataFrame([])

    data = pd.DataFrame(data)  # cast to dataframe

    cdata = pd.DataFrame(index=data.index)

    for k in observables:
        # Protect against missing data
        if k not in data:
            cdata[prefix + k] = np.inf
            continue

        ovalues = data[k]
        cdata[prefix + k] = ovalues.fillna(np.inf)  # Protect against nans
    return cdata


def form_feasibility_data(constraints: Dict, data, prefix="feasible_"):
    """
    Use constraint dict and data to identify feasible points in the the dataset.

    Returns a dataframe with the feasible data.
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


def validate_input_data(vocs, data):
    for name in vocs.variable_names:
        lower = vocs.variables[name][0]
        upper = vocs.variables[name][1]

        d = data[name]

        # see if points violate limits
        is_out_of_bounds = pd.DataFrame((d < lower, d > upper)).any(axis=0)
        is_out_of_bounds_idx = list(is_out_of_bounds[is_out_of_bounds].index)

        if len(is_out_of_bounds_idx):
            raise ValueError(
                f"input points at indices {is_out_of_bounds_idx} are not valid for {name} range in VOCS!"
            )
