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
        return list(sorted(self.constraints.keys()))

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
        if include_constants:
            inputs.update(self.constants)

        # Handle linked variables
        if self.linked_variables:
            for k, v in self.linked_variables.items():
                inputs[k] = inputs[v]

        return inputs

    def convert_dataframe_to_inputs(self, inputs: pd.DataFrame) -> List[Dict]:
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

        return in_copy.to_dict("records")

    def convert_numpy_to_inputs(self, inputs: np.ndarray) -> List[Dict]:
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
        inputs = data[self.variable_names].to_numpy()
        outputs = data[self.objective_names + self.constraint_names].to_numpy()

        return inputs, outputs

    def append_constraints(self, data: pd.DataFrame):
        """
        transform constraints from dataframe to imply feasibility if value is < 0
        according to vocs
        """
        for name, value in self.constraints.items():
            if value[0] == "GREATER_THAN":
                data[f"{name}_f"] = -(data[name] - value[1])
            else:
                data[f"{name}_f"] = data[name] - value[1]

        # add feasibility metric if all values are <= 0
        data["feasibility"] = (
            data[[f"{ele}_f" for ele in self.constraint_names]] <= 0
        ).all(axis=1)
        return data





# --------------------------------
# dataframe utilities

NAN_CONST = -666
OBJECTIVE_WEIGHT = {'MINIMIZE': 1.0, 'MAXIMIZE': -1.0}


def objective_data(vocs, data, prefix='objective_'):
    """
    Use objective dict and data (dataframe) to generate objective data (dataframe)

    Weights are applied to convert all objectives into mimimization form.

    Returns a dataframe with the objective data intented to be minimized.

    """
    data = pd.DataFrame(data)
    objective_dict = vocs.objectives

    odata = pd.DataFrame()
    for k in sorted(list(objective_dict)):
        operator = objective_dict[k].upper()
        if operator not in OBJECTIVE_WEIGHT:
            raise ValueError(f'Unknown objective operator: {operator}')

        weight = OBJECTIVE_WEIGHT[operator]
        odata[prefix + k] = weight*data[k]
        
    return odata

def constraint_data(vocs, data, prefix='constraint_'):
    """
    Use constraint dict and data (dataframe) to generate constraint data (dataframe)
    A constraint is satisfied if the evaluation is > 0.

    Returns a dataframe with the constraint data.
    """

    data = pd.DataFrame(data) # cast to dataframe
    constraint_dict = vocs.constraints

    cdata = pd.DataFrame()
    for k in sorted(list(constraint_dict)):
        x = data[k]
        op, d = constraint_dict[k]
        op = op.upper()  # Allow any case

        if op == 'GREATER_THAN':  # x > d -> x-d > 0
            cvalues = (x - d)
        elif op == 'LESS_THAN':  # x < d -> d-x > 0
            cvalues = (d - x)
        else:
            raise ValueError(f'Unknown constraint operator: {op}')

        cdata[prefix+k] = cvalues.fillna(NAN_CONST)

    return cdata