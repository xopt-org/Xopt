import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms import Bilog, Normalize, Standardize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior
from torch import Tensor

from xopt.generators.bayesian.models.base import ModelConstructor
from xopt.vocs import VOCS


class StandardModel(ModelConstructor):
    def __init__(self, vocs: VOCS, tkwargs, use_low_noise_prior=False):
        super().__init__(vocs, tkwargs)
        self.use_low_noise_prior = use_low_noise_prior

        # get input transform
        self.input_transform = Normalize(
            self.vocs.n_variables, bounds=torch.tensor(self.vocs.bounds, **self.tkwargs)
        )

        # get likelihood
        self.likelihood = None
        if self.use_low_noise_prior:
            self.likelihood = GaussianLikelihood(noise_prior=GammaPrior(1.0, 10.0))

    def build_model(self, data: pd.DataFrame) -> ModelListGP:
        """construct independent model for each objective and constraint"""

        # get data
        input_data, objective_data, constraint_data = self.vocs.extract_data(data)
        train_X = torch.tensor(input_data.to_numpy(), **self.tkwargs)

        models = []
        for name in objective_data.keys():
            train_Y = torch.tensor(
                objective_data[name].to_numpy(), **self.tkwargs
            ).unsqueeze(-1)
            models.append(self.build_objective_model(train_X, train_Y))

        for name in constraint_data.keys():
            train_C = torch.tensor(
                constraint_data[name].to_numpy(), **self.tkwargs
            ).unsqueeze(-1)
            models.append(self.build_objective_model(train_X, train_C))

        return ModelListGP(*models)

    def build_objective_model(self, train_X: Tensor, train_Y: Tensor) -> Model:
        """construct model for each objective"""
        outcome_transform = Standardize(1)

        return self._build_single_task_gp(
            train_X,
            train_Y,
            input_transform=self.input_transform,
            outcome_transform=outcome_transform,
            likelihood=self.likelihood,
        )

    def build_constraint_model(self, train_X: Tensor, train_Y: Tensor) -> Model:
        """construct model for each constraint"""
        outcome_transform = Bilog()

        return self._build_single_task_gp(
            train_X,
            train_Y,
            input_transform=self.input_transform,
            outcome_transform=outcome_transform,
            likelihood=self.likelihood,
        )


def create_standard_model(
    data,
    vocs,
    use_conservative_prior_lengthscale: bool = False,
    use_conservative_prior_mean: bool = False,
    use_low_noise_prior: bool = False,
):
    input_data, objective_data, constraint_data = vocs.extract_data(data)
    tkwargs = {"dtype": torch.double, "device": "cpu"}

    input_transform = Normalize(
        vocs.n_variables, bounds=torch.tensor(vocs.bounds, **tkwargs)
    )

    objective_models = create_objective_models(
        input_data, objective_data, input_transform, tkwargs, use_low_noise_prior
    )
    constraint_models = create_constraint_models(
        input_data,
        constraint_data,
        input_transform,
        tkwargs,
        use_low_noise_prior,
        use_conservative_prior_lengthscale,
        use_conservative_prior_mean,
    )

    return ModelListGP(*objective_models, *constraint_models)


def create_objective_models(
    input_data, objective_data, input_transform, tkwargs, use_low_noise_prior=False
):
    # validate data
    if len(input_data) == 0:
        raise ValueError("input_data is empty/all Nans, cannot create model")

    train_X = torch.tensor(input_data.to_numpy(), **tkwargs)

    models = []

    # create models for objectives
    for name in objective_data.keys():
        train_Y = torch.tensor(objective_data[name].to_numpy(), **tkwargs).unsqueeze(-1)

        likelihood = None
        if use_low_noise_prior:
            likelihood = GaussianLikelihood(noise_prior=GammaPrior(1.0, 10.0))

        models.append(
            SingleTaskGP(
                train_X,
                train_Y,
                input_transform=input_transform,
                outcome_transform=Standardize(1),
                likelihood=likelihood,
            )
        )
        mll = ExactMarginalLogLikelihood(models[-1].likelihood, models[-1])
        fit_gpytorch_mll(mll)

    return models


def create_constraint_models(
    input_data,
    constraint_data,
    input_transform,
    tkwargs,
    use_low_noise_prior=False,
    use_conservative_prior_lengthscale: bool = False,
    use_conservative_prior_mean: bool = False,
):
    # validate data
    if len(input_data) == 0:
        raise ValueError("input_data is empty/all Nans, cannot create model")

    train_X = torch.tensor(input_data.to_numpy(), **tkwargs)
    models = []

    # do constraint models
    for name in constraint_data.keys():
        train_Y = torch.tensor(constraint_data[name].to_numpy(), **tkwargs).unsqueeze(
            -1
        )

        outcome_transform = Bilog()

        # use conservative priors if requested
        covar_module = None
        if use_conservative_prior_lengthscale:
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=1.5,
                    ard_num_dims=train_X.shape[-1],
                    lengthscale_prior=GammaPrior(10.0, 100.0),
                ),
                outputscale_prior=GammaPrior(5.0, 1.0),
            )

        likelihood = None
        if use_low_noise_prior:
            likelihood = GaussianLikelihood(noise_prior=GammaPrior(1.0, 10.0))

        models.append(
            SingleTaskGP(
                train_X,
                train_Y,
                input_transform=input_transform,
                outcome_transform=outcome_transform,
                covar_module=covar_module,
                likelihood=likelihood,
            )
        )

        if use_conservative_prior_mean:
            models[-1].mean_module.constant.data = torch.tensor(1.0, **tkwargs)
            models[-1].mean_module.constant.requires_grad = False

        mll = ExactMarginalLogLikelihood(models[-1].likelihood, models[-1])
        fit_gpytorch_mll(mll)

    # create model list
    return models
