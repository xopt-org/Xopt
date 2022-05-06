# globally modify pydantic base model to not allow extra keys
from pydantic import BaseModel


class XoptBaseModel(BaseModel):
    class Config:
        extra = 'forbid'
