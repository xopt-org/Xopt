import os.path
from typing import Dict, Optional

import pandas as pd
import torch
from botorch.models import ModelListGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior
from pydantic import Field, validator
from torch.nn import Module

from xopt.generators.bayesian.base_model import ModelConstructor
from xopt.generators.bayesian.models.prior_mean import CustomMean
from xopt.pydantic import orjson_dumps

DECODERS = {"torch.float32": torch.float32, "torch.float64": torch.float64}


class StandardModelConstructor(ModelConstructor):
    name = "standard"
    use_low_noise_prior: bool = Field(
        True, description="specify if model should assume a low noise environment"
    )
    covar_modules: Dict[str, Kernel] = Field(
        {}, description="covariance modules for GP models"
    )
    mean_modules: Dict[str, Module] = Field(
        {}, description="prior mean modules for GP models"
    )
    dtype: torch.dtype = Field(torch.double)
    device: str = Field("cpu")

    class Config:
        arbitrary_types_allowed = True
        json_dumps = orjson_dumps
        validate_assignment = True

    @validator("covar_modules", "mean_modules", pre=True)
    def validate_torch_modules(cls, v):
        if not isinstance(v, dict):
            raise ValueError("must be dict")
        else:
            for key, val in v.items():
                if isinstance(val, str):
                    if os.path.exists(val):
                        v[key] = torch.load(val)

        return v

    @validator("dtype", pre=True)
    def validate_dtype(cls, v):
        if isinstance(v, str):
            try:
                return DECODERS[v]
            except KeyError:
                raise ValueError(f"cannot convert {v}")
        return v

    @property
    def input_transform(self):
        return Normalize(
            self.vocs.n_variables, bounds=torch.tensor(self.vocs.bounds)
        ).to(**self.tkwargs)

    @property
    def likelihood(self):
        if self.use_low_noise_prior:
            return GaussianLikelihood(noise_prior=GammaPrior(1.0, 10.0))
        else:
            return GaussianLikelihood()

    @property
    def tkwargs(self):
        return {"dtype": self.dtype, "device": self.device}

    def build_model(self, data: pd.DataFrame) -> ModelListGP:
        """construct independent model for each objective and constraint"""

        # build model
        pd.options.mode.use_inf_as_na = True
        models = []
        for name in self.vocs.output_names:
            outcome_transform = Standardize(1)
            covar_module = self._get_module(self.covar_modules, name)
            mean_module = self.build_mean_module(name, outcome_transform)
            train_X, train_Y = self._get_training_data(name, data)
            models.append(
                self.build_single_task_gp(
                    train_X,
                    train_Y,
                    input_transform=self.input_transform,
                    outcome_transform=outcome_transform,
                    covar_module=covar_module,
                    mean_module=mean_module,
                    likelihood=self.likelihood,
                )
            )
        return ModelListGP(*models)

    def build_mean_module(self, name, outcome_transform) -> Optional[CustomMean]:
        """Builds the mean module for the output specified by name."""
        mean_module = self._get_module(self.mean_modules, name)
        if mean_module is not None:
            mean_module = CustomMean(
                mean_module, self.input_transform, outcome_transform
            )
        return mean_module

    def _get_training_data(self, name, data) -> (torch.Tensor, torch.Tensor):
        """Returns (train_X, train_Y) for the output specified by name."""
        # collect data from dataframe
        input_data, objective_data, constraint_data = self.vocs.extract_data(
            data, return_raw=True
        )

        # objective specifics
        if name in self.vocs.objective_names:
            target_data = objective_data
        # constraint specifics
        elif name in self.vocs.constraint_names:
            target_data = constraint_data
        else:
            raise RuntimeError(
                "Output '{}' is not found in either objectives or "
                "constraints.".format(name)
            )
        train_X = torch.tensor(
            input_data[~target_data[name].isnull()].to_numpy(), **self.tkwargs
        )
        train_Y = torch.tensor(
            target_data[~target_data[name].isnull()][name].to_numpy(), **self.tkwargs
        ).unsqueeze(-1)
        return train_X, train_Y

    @staticmethod
    def _get_module(base, name):
        if isinstance(base, Module):
            return base
        elif isinstance(base, dict):
            return base.get(name)
        else:
            return None


class _NegativeModule(Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, X):
        return -self.base(X)
