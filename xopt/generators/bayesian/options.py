from typing import Dict, List

import torch

from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.sampling import MCSampler, SobolQMCNormalSampler
from pydantic import BaseModel
from torch import Tensor


class AcqOptions(BaseModel):
    # objective creation
    objective: MCAcquisitionObjective = None

    # monte carlo options
    sampler: MCSampler = SobolQMCNormalSampler(num_samples=512)

    class Config:
        arbitrary_types_allowed = True


class OptimOptions(BaseModel):
    num_restarts: int = 5
    raw_samples: int = 20
    sequential: bool = True


class ModelOptions(BaseModel):
    input_transform: InputTransform = None
    outcome_transform: OutcomeTransform = None

    class Config:
        arbitrary_types_allowed = True


class BayesianOptions(BaseModel):
    optim: OptimOptions = OptimOptions()
    acq: AcqOptions = AcqOptions()
    model: ModelOptions = ModelOptions()
    tkwargs: Dict = {"device": "cpu", "dtype": torch.double}
    n_initial: int = 3
    proximal_lengthscales: List[float] = None

    def update(self, **kwargs):
        all_kwargs = kwargs

        def set_recursive(d: BaseModel):
            if not isinstance(d, dict):
                for name, val in d.__fields__.items():
                    attr = getattr(d, name)
                    if isinstance(attr, BaseModel):
                        set_recursive(attr)
                    elif name in kwargs.keys():
                        setattr(d, name, all_kwargs.pop(name))
                    else:
                        pass

        set_recursive(self)

        if len(all_kwargs):
            raise RuntimeError(
                f"keys {list(all_kwargs.keys())} not found, will not be " f"updated!"
            )


if __name__ == "__main__":
    options = BayesianOptions()
    options.optim.raw_samples = 30
    print(options.dict())
