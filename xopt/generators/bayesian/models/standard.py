import os.path
from copy import deepcopy
from typing import Dict, List, Optional

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
    def likelihood(self):
        if self.use_low_noise_prior:
            return GaussianLikelihood(
                noise_prior=GammaPrior(1.0, 100.0),
            )
        else:
            return GaussianLikelihood()

    @property
    def tkwargs(self):
        return {"dtype": self.dtype, "device": self.device}

    def get_input_transform(
        self, input_names: List, input_bounds: Dict[str, List] = None
    ):
        if input_bounds is None:
            bounds = None
        else:
            bounds = torch.vstack(
                [torch.tensor(input_bounds[name]) for name in input_names]
            ).T
        return Normalize(len(input_names), bounds=bounds).to(**self.tkwargs)

    def build_model(
        self,
        input_names: List[str],
        outcome_names: List[str],
        data: pd.DataFrame,
        input_bounds: Dict[str, List] = None,
    ) -> ModelListGP:
        """construct independent model for each objective and constraint"""

        # build model
        pd.options.mode.use_inf_as_na = True
        models = []
        input_transform = self.get_input_transform(input_names, input_bounds)

        for name in outcome_names:
            outcome_transform = Standardize(1)
            covar_module = self._get_module(self.covar_modules, name)
            mean_module = self.build_mean_module(
                name, input_transform, outcome_transform
            )
            train_X, train_Y = self._get_training_data(input_names, name, data)
            models.append(
                self.build_single_task_gp(
                    train_X,
                    train_Y,
                    input_transform=input_transform,
                    outcome_transform=outcome_transform,
                    covar_module=covar_module,
                    mean_module=mean_module,
                    likelihood=self.likelihood,
                )
            )
        return ModelListGP(*models)

    def build_mean_module(
        self, name, input_transform, outcome_transform
    ) -> Optional[CustomMean]:
        """Builds the mean module for the output specified by name."""
        mean_module = self._get_module(self.mean_modules, name)
        if mean_module is not None:
            mean_module = CustomMean(mean_module, input_transform, outcome_transform)
        return mean_module

    def _get_training_data(
        self, input_names: List[str], outcome_name: str, data: pd.DataFrame
    ) -> (torch.Tensor, torch.Tensor):
        """Returns (train_X, train_Y) for the output specified by name."""
        input_data = data[input_names]
        outcome_data = data[outcome_name]

        # cannot use any rows where any variable values are nans
        non_nans = ~input_data.isnull().T.any()
        input_data = input_data[non_nans]
        outcome_data = outcome_data[non_nans]

        train_X = torch.tensor(
            input_data[~outcome_data.isnull()].to_numpy(), **self.tkwargs
        )
        train_Y = torch.tensor(
            outcome_data[~outcome_data.isnull()].to_numpy(), **self.tkwargs
        ).unsqueeze(-1)
        return train_X, train_Y

    @staticmethod
    def _get_module(base, name):
        if isinstance(base, Module):
            return deepcopy(base)
        elif isinstance(base, dict):
            return deepcopy(base.get(name))
        else:
            return None
