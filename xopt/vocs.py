from pydantic import BaseModel, conlist
from enum import Enum
from typing import Dict, Union, List, Tuple, Any
import yaml


# Enums for objectives and constraints
class ObjectiveEnum(str, Enum):
    MINIMIZE = 'MINIMIZE'
    MAXIMIZE = 'MAXIMIZE'
    
    # Allow any case
    @classmethod
    def _missing_(cls, name):
        for member in cls:
            if member.name.lower() == name.lower():
                return member    
    
class ConstraintEnum(str, Enum):
    LESS_THAN = 'LESS_THAN'
    GREATER_THAN = 'GREATER_THAN'    
    
    # Allow any case
    @classmethod
    def _missing_(cls, name):
        for member in cls:
            if member.name.lower() == name.lower():
                return member
            
class VOCS(BaseModel):
    variables: Dict[str, conlist(float, min_items=2, max_items=2)] = None
    constraints: Dict[str, conlist(Union[float, ConstraintEnum], min_items=2, max_items=2)] = None
    objectives: Dict[str, ObjectiveEnum] = None
    constants: Dict[str, Any] = None
    linked_variables: Dict[str, str] = None
        
    class Config:
        validate_assignment=True # Not sure this helps in this case
        use_enum_values = True 
    
    @classmethod
    def from_yaml(cls, yaml_text):
        return cls.parse_obj(yaml.safe_load(yaml_text))
    
    def as_yaml(self):
        return yaml.dump(self.dict(), default_flow_style=None, sort_keys=False)    