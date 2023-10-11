import os.path
from copy import deepcopy
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from botorch.models import ModelListGP
from botorch.models.transforms import Standardize
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior
from pydantic import ConfigDict, Field, field_validator
from pydantic_core.core_schema import FieldValidationInfo
from torch.nn import Module

from xopt.generators.bayesian.base_model import ModelConstructor
from xopt.generators.bayesian.models.prior_mean import CustomMean
from xopt.generators.bayesian.utils import get_input_transform, get_training_data
from xopt.pydantic import decode_torch_module

DECODERS = {"torch.float32": torch.float32, "torch.float64": torch.float64}
MIN_INFERRED_NOISE_LEVEL = 1e-4


class StandardModelConstructor(ModelConstructor):
    name: str = Field("standard", frozen=True)
    use_low_noise_prior: bool = Field(
        True, description="specify if model should assume a low noise environment"
    )
    covar_modules: Dict[str, Kernel] = Field(
        {}, description="covariance modules for GP models"
    )
    mean_modules: Dict[str, Module] = Field(
        {}, description="prior mean modules for GP models"
    )
    trainable_mean_keys: List[str] = Field(
        [], description="list of prior mean modules that can be trained"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @field_validator("covar_modules", "mean_modules", mode="before")
    def validate_torch_modules(cls, v):
        if not isinstance(v, dict):
            raise ValueError("must be dict")
        else:
            for key, val in v.items():
                if isinstance(val, str):
                    if val.startswith("base64:"):
                        v[key] = decode_torch_module(val)
                    elif os.path.exists(val):
                        v[key] = torch.load(val)

        return v

    @field_validator("trainable_mean_keys")
    def validate_trainable_mean_keys(cls, v, info: FieldValidationInfo):
        for name in v:
            assert name in info.data["mean_modules"]
        return v

    @property
    def likelihood(self):
        if self.use_low_noise_prior:
            return GaussianLikelihood(
                noise_prior=GammaPrior(1.0, 100.0),
            )
        else:
            noise_prior = GammaPrior(1.1, 0.05)
            noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
            likelihood = GaussianLikelihood(
                noise_prior=noise_prior,
                noise_constraint=GreaterThan(
                    MIN_INFERRED_NOISE_LEVEL,
                    transform=None,
                    initial_value=noise_prior_mode,
                ),
            )
            return likelihood

    def build_model(
        self,
        input_names: List[str],
        outcome_names: List[str],
        data: pd.DataFrame,
        input_bounds: Dict[str, List] = None,
        dtype: torch.dtype = torch.double,
        device: Union[torch.device, str] = "cpu",
    ) -> ModelListGP:
        """construct independent model for each objective and constraint"""

        # build model
        tkwargs = {"dtype": dtype, "device": device}
        models = []
        input_transform = get_input_transform(input_names, input_bounds).to(**tkwargs)

        for name in outcome_names:
            outcome_transform = Standardize(1)
            covar_module = self._get_module(self.covar_modules, name)
            mean_module = self.build_mean_module(
                name, input_transform, outcome_transform
            )

            train_X, train_Y, train_Yvar = get_training_data(input_names, name, data)
            kwargs = {
                "input_transform": input_transform,
                "outcome_transform": outcome_transform,
                "covar_module": covar_module,
                "mean_module": mean_module,
            }

            if train_Yvar is None:
                # train basic single-task-gp model
                models.append(
                    self.build_single_task_gp(
                        train_X.to(**tkwargs),
                        train_Y.to(**tkwargs),
                        likelihood=self.likelihood,
                        **kwargs
                    )
                )
            else:
                # train heteroskedastic single-task-gp model
                # turn off warnings
                models.append(
                    self.build_heteroskedastic_gp(
                        train_X.to(**tkwargs),
                        train_Y.to(**tkwargs),
                        train_Yvar.to(**tkwargs),
                        **kwargs
                    )
                )

        return ModelListGP(*models)

    def build_mean_module(
        self, name, input_transform, outcome_transform
    ) -> Optional[CustomMean]:
        """Builds the mean module for the output specified by name."""
        mean_module = self._get_module(self.mean_modules, name)
        if mean_module is not None:
            fixed_model = False if name in self.trainable_mean_keys else True
            mean_module = CustomMean(
                mean_module, input_transform, outcome_transform, fixed_model=fixed_model
            )
        return mean_module

    @staticmethod
    def _get_module(base, name):
        if isinstance(base, Module):
            return deepcopy(base)
        elif isinstance(base, dict):
            return deepcopy(base.get(name))
        else:
            return None
