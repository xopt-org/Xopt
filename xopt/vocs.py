import numpy as np
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
    def bounds(self ):
        """
        Returns a bounds array (mins, maxs) of shape (2, n_variables)
        Arrays of lower and upper bounds can be extracted by:
            mins, maxs = vocs.bounds
        """
        return np.array([v for _, v in sorted(self.variables.items())]).T        
        #return np.vstack([np.array(ele) for _, ele in self.variables.items()]).T

    def random_inputs(self, n=None, include_constants=True, include_linked_variables=True):
        """
        Uniform sampling of the variables.

        Returns a dict of inputs. 
        
        If include_constants, the vocs.constants are added to the dict. 

        Optional:
            n (integer) to make arrays of inputs, of size n. 
        
        """
        inputs = {}
        for key, val in self.variables.items(): # No need to sort here
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

