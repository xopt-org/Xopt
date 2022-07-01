import pandas as pd
import torch
from botorch import fit_gpytorch_model
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import ChainedOutcomeTransform, Normalize, Standardize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior

from xopt.generators.bayesian.custom_botorch.bilog import Bilog
from xopt.generators.bayesian.custom_botorch.constraint_transform import Constraint

from xopt.vocs import VOCS


def create_standard_model(
    data: pd.DataFrame,
    vocs: VOCS,
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
    data = data[vocs.variable_names + vocs.output_names].dropna()

    train_X = torch.tensor(data[vocs.variable_names].to_numpy(), **tkwargs)
    bounds = torch.tensor(vocs.bounds, **tkwargs)
    normalize = Normalize(vocs.n_variables, bounds=bounds)

    models = {}
    for name in vocs.objective_names:
        train_Y = torch.tensor(data[name].to_numpy(), **tkwargs).unsqueeze(-1)

        likelihood = None
        if use_low_noise_prior:
            likelihood = GaussianLikelihood(noise_prior=GammaPrior(1.0, 10.0))

        models[name] = SingleTaskGP(
            train_X,
            train_Y,
            input_transform=normalize,
            outcome_transform=Standardize(1),
            likelihood=likelihood,
        )
        mll = ExactMarginalLogLikelihood(models[name].likelihood, models[name])
        fit_gpytorch_model(mll)

    # do constraint models
    for name, val in vocs.constraints.items():
        train_Y = torch.tensor(data[name].to_numpy(), **tkwargs).unsqueeze(-1)

        outcome_transform = ChainedOutcomeTransform(
            constraint=Constraint({0: vocs.constraints[name]}),
            bilog=Bilog(m=1),
        )

        # use conservative priors if requested
        covar_module = None
        if use_conservative_prior_lengthscale:
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=1.5,
                    ard_num_dims=vocs.n_variables,
                    lengthscale_prior=GammaPrior(10.0, 100.0),
                ),
                outputscale_prior=GammaPrior(5.0, 1.0),
            )

        likelihood = None
        if use_low_noise_prior:
            likelihood = GaussianLikelihood(noise_prior=GammaPrior(1.0, 10.0))

        models[name] = SingleTaskGP(
            train_X,
            train_Y,
            input_transform=normalize,
            outcome_transform=outcome_transform,
            covar_module=covar_module,
            likelihood=likelihood,
        )

        if use_conservative_prior_mean:
            models[name].mean_module.constant = torch.nn.Parameter(
                torch.tensor(5.0, **tkwargs), requires_grad=False
            )

        mll = ExactMarginalLogLikelihood(models[name].likelihood, models[name])
        fit_gpytorch_model(mll)

    # create model list
    return ModelListGP(*[models[name] for name in vocs.output_names])
