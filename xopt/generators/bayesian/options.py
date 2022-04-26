from typing import Dict, List

import torch
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.sampling import MCSampler, SobolQMCNormalSampler
from pydantic import BaseModel

from xopt.generator import GeneratorOptions


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


class BayesianOptions(GeneratorOptions):
    optim: OptimOptions = OptimOptions()
    acq: AcqOptions = AcqOptions()
    model: ModelOptions = ModelOptions()
    tkwargs: Dict = {"device": "cpu", "dtype": torch.double}
    n_initial: int = 3
    proximal_lengthscales: List[float] = None




if __name__ == "__main__":
    options = BayesianOptions()
    options.optim.raw_samples = 30
    print(options.dict())
