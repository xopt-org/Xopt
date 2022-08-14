import pandas as pd
import torch
from botorch import fit_gpytorch_model
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Bilog, Normalize, Standardize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior


def create_standard_model(
    input_data: pd.DataFrame,
    objective_data: pd.DataFrame,
    constraint_data: pd.DataFrame,
    bounds,
    use_conservative_prior_lengthscale: bool = False,
    use_conservative_prior_mean: bool = False,
    use_low_noise_prior: bool = False,
    tkwargs: dict = None,
) -> ModelListGP:
    """
    Generate a standard ModelListGP for use in optimization
    - model inputs are normalized to the [0,1] domain
    - model outputs are transformed according to output type
        - objectives are standardized to have zero mean and unit standard deviation
        - constraints are transformed according to `vocs` such that negative values
            imply feasibility and extreme values are damped using a Bilog transform (
            see (https://arxiv.org/abs/2002.08526) for details
    """
    tkwargs = tkwargs or {"dtype": torch.double, "device": "cpu"}

    # validate data
    if len(input_data) == 0:
        raise ValueError("input_data is empty/all Nans, cannot create model")

    train_X = torch.tensor(input_data.to_numpy(), **tkwargs)
    bounds = torch.tensor(bounds, **tkwargs)
    normalize = Normalize(train_X.shape[-1], bounds=bounds)

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
                input_transform=normalize,
                outcome_transform=Standardize(1),
                likelihood=likelihood,
            )
        )
        mll = ExactMarginalLogLikelihood(models[-1].likelihood, models[-1])
        fit_gpytorch_model(mll)

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
                input_transform=normalize,
                outcome_transform=outcome_transform,
                covar_module=covar_module,
                likelihood=likelihood,
            )
        )

        if use_conservative_prior_mean:
            models[-1].mean_module.constant.data = torch.tensor(5.0, **tkwargs)
            models[-1].mean_module.constant.requires_grad = False

        mll = ExactMarginalLogLikelihood(models[-1].likelihood, models[-1])
        fit_gpytorch_model(mll)

    # create model list
    return ModelListGP(*models)
