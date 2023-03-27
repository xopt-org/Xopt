import pandas as pd
import torch
from botorch.models import ModelListGP
from botorch.models.transforms import Bilog, Normalize, Standardize
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior
from torch import Tensor

from xopt.generators.bayesian.base_model import ModelConstructor
from xopt.vocs import VOCS


class StandardModelConstructor(ModelConstructor):
    def __init__(self, vocs: VOCS, options):
        super().__init__(vocs, options)
        # get input transform
        self.input_transform = Normalize(
            self.vocs.n_variables, bounds=torch.tensor(self.vocs.bounds)
        )

        # get likelihood
        self.likelihood = None
        if self.options.use_low_noise_prior:
            self.likelihood = GaussianLikelihood(noise_prior=GammaPrior(1.0, 10.0))

    def build_model(self, data: pd.DataFrame, tkwargs: dict = None) -> ModelListGP:
        """construct independent model for each objective and constraint"""
        # set tkwargs
        tkwargs = tkwargs or {"dtype": torch.double, "device": "cpu"}

        # drop nans
        valid_data = data[
            pd.unique(self.vocs.variable_names + self.vocs.output_names)
        ].dropna()

        # get data
        input_data, objective_data, constraint_data = self.vocs.extract_data(valid_data)
        train_X = torch.tensor(input_data.to_numpy(), **tkwargs)
        self.input_transform.to(**tkwargs)
        self.likelihood.to(**tkwargs)

        return self.build_standard_model(
            train_X, objective_data, constraint_data, tkwargs
        )

    def build_standard_model(
        self,
        train_X: Tensor,
        objective_data: pd.DataFrame,
        constraint_data: pd.DataFrame,
        tkwargs,
        **model_kwargs
    ):
        models = []
        for name in objective_data.keys():
            train_Y = torch.tensor(
                objective_data[name].to_numpy(), **tkwargs
            ).unsqueeze(-1)
            outcome_transform = Standardize(1)
            models.append(
                self._build_single_task_gp(
                    train_X,
                    train_Y,
                    input_transform=self.input_transform,
                    outcome_transform=outcome_transform,
                    likelihood=self.likelihood,
                    **model_kwargs
                )
            )

        for name in constraint_data.keys():
            train_C = torch.tensor(
                constraint_data[name].to_numpy(), **tkwargs
            ).unsqueeze(-1)
            outcome_transform = Bilog()
            models.append(
                self._build_single_task_gp(
                    train_X,
                    train_C,
                    input_transform=self.input_transform,
                    outcome_transform=outcome_transform,
                    likelihood=self.likelihood,
                    **model_kwargs
                )
            )

        return ModelListGP(*models)
