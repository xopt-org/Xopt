# globally modify pydantic base model to not allow extra keys and handle np arrays
import numpy as np
from pydantic import BaseModel


class XoptBaseModel(BaseModel):
    class Config:
        extra = 'forbid'
        json_encoders = {
            np.ndarray: lambda x: x.tolist(),
            np.int64: lambda x: int(x),
            np.float64: lambda x: float(x),
        }
