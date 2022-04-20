import numpy as np
import pandas as pd
from pydantic import BaseModel, conlist
from enum import Enum
from typing import Dict, Union, List, Tuple, Any
import yaml


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


class VOCS(BaseModel):
    variables: Dict[str, conlist(float, min_items=2, max_items=2)] = None
    constraints: Dict[
        str, conlist(Union[float, ConstraintEnum], min_items=2, max_items=2)
    ] = None
    objectives: Dict[str, ObjectiveEnum] = None
    constants: Dict[str, Any] = None
    linked_variables: Dict[str, str] = None

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
        return list(sorted(self.variables.keys()))

    @property
    def objective_names(self):
        return list(sorted(self.objectives.keys()))

    @property
    def constraint_names(self):
        if self.constraints is None:
            return []
        return list(sorted(self.constraints.keys()))

    @property
    def constant_names(self):
        if self.constants is None:
            return []
        return list(sorted(self.constants.keys()))

    @property
    def all_names(self):
        return (
            self.variable_names
            + self.constant_names
            + self.objective_names
            + self.constraint_names
        )

    @property
    def n_variables(self):
        return len(self.variables)

    @property
    def n_constants(self):
        return len(self.constants)

    @property
    def n_inputs(self):
        return self.n_variables + self.n_constants

    @property
    def n_objectives(self):
        return len(self.objectives)

    @property
    def n_constraints(self):
        return len(self.constraints)

    @property
    def n_outputs(self):
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

    def convert_dataframe_to_inputs(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a dataframe candidate locations to a
        list of dicts to pass to executors.
        """
        # make sure that the df keys contain the vocs variables
        if not set(self.variable_names).issubset(set(inputs.keys())):
            raise RuntimeError(
                f"input dataframe must at least contain the vocs " f"variables"
            )

        in_copy = inputs.copy()

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

    def get_training_data(self, data: pd.DataFrame):
        """
        get training data from dataframe (usually supplied by xopt base)

        """
        inputs = data[self.variable_names].to_numpy(np.float64)
        outputs = data[self.objective_names + self.constraint_names].to_numpy(
            np.float64
        )

        return inputs, outputs

    # Extract optimization data (in correct column order)
    def variable_data(self, data, prefix='variable_'):
        return form_variable_data(self.variables, data, prefix=prefix)

    def objective_data(self, data, prefix="objective_"):
        return form_objective_data(self.objectives, data, prefix)

    def constraint_data(self, data, prefix="constraint_"):     
        return form_constraint_data(self.constraints, data, prefix)

    def feasibility_data(self, data, prefix="feasible_"):
        return form_feasibility_data(self.constraints, data, prefix)


# --------------------------------
# dataframe utilities

OBJECTIVE_WEIGHT = {"MINIMIZE": 1.0, "MAXIMIZE": -1.0}


def form_variable_data(variables: Dict, data, prefix='variable_'):
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

    """
    if not objectives:
        return None

    data = pd.DataFrame(data)

    odata = pd.DataFrame()
    for k in sorted(list(objectives)):
        operator = objectives[k].upper()
        if operator not in OBJECTIVE_WEIGHT:
            raise ValueError(f"Unknown objective operator: {operator}")

        weight = OBJECTIVE_WEIGHT[operator]
        odata[prefix + k] = weight * data[k]

    return odata


def form_constraint_data(constraints: Dict, data, prefix="constraint_"):
    """
    Use constraint dict and data (dataframe) to generate constraint data (dataframe)
    A constraint is satisfied if the evaluation is < 0.

    Returns a dataframe with the constraint data.
    """
    if not constraints:
        return None

    data = pd.DataFrame(data)  # cast to dataframe
    constraint_dict = constraints

    cdata = pd.DataFrame()
    for k in sorted(list(constraints)):
        x = data[k]
        op, d = constraints[k]
        op = op.upper()  # Allow any case

        if op == "GREATER_THAN":  # x > d -> x-d > 0
            cvalues = -(x - d)
        elif op == "LESS_THAN":  # x < d -> d-x > 0
            cvalues = -(d - x)
        else:
            raise ValueError(f"Unknown constraint operator: {op}")

        cdata[prefix + k] = cvalues
    return cdata


def form_feasibility_data(constraints: Dict, data, prefix="feasible_"):
    """
    Use constraint dict and data to identify feasible points in the the dataset.

    Returns a dataframe with the feasibility data.
    """
    if not constraints:
        df = pd.DataFrame(index=data.index)
        df['feasible'] = True
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
