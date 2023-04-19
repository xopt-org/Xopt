import torch
from botorch import fit_gpytorch_model
from botorch.models import ModelListGP, SingleTaskGP

from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, PolynomialKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import GammaPrior


def create_emittance_model(data, vocs, meas_param, noise="low"):
    # data will be a pandas dataframe
    tkwargs = {"dtype": torch.double}
    input_data, objective_data, constraint_data = vocs.extract_data(data)

    train_X = torch.tensor(input_data.to_numpy(), **tkwargs)

    # covar
    meas_dim = input_data.columns.get_loc(meas_param)
    tuning_dims = list(range(vocs.n_variables))
    tuning_dims.remove(meas_dim)
    covar_module = MaternKernel(active_dims=tuning_dims) * PolynomialKernel(
        power=2, active_dims=[meas_dim]
    )
    scaled_covar_module = ScaleKernel(covar_module)

    # mean
    constant_constraint = None
    constant_prior = None
    mean_module = ConstantMean(
        constant_prior=constant_prior, constant_constraint=constant_constraint
    )

    # noise/likelihood
    if noise == "low":
        noise_prior = GammaPrior(1, 10)
    else:  # could add more options but for now I always use GammaPrior(1,10)
        noise_prior = None
    noise_constraint = None
    likelihood = GaussianLikelihood(
        noise_prior=noise_prior, noise_constraint=noise_constraint
    )

    #     transforms
    input_transform = Normalize(
        vocs.n_variables, bounds=torch.tensor(vocs.bounds, **tkwargs)
    )

    objective_models = []

    # create models for objective(s)
    for name in objective_data.keys():
        train_Y = torch.tensor(objective_data[name].to_numpy(), **tkwargs).unsqueeze(-1)

        objective_models.append(
            SingleTaskGP(
                train_X,
                train_Y,
                input_transform=input_transform,
                outcome_transform=Standardize(1),
                likelihood=likelihood,
                covar_module=scaled_covar_module,
                mean_module=mean_module,
            )
        )
        mll = ExactMarginalLogLikelihood(
            objective_models[-1].likelihood, objective_models[-1]
        )
        fit_gpytorch_model(mll)

    return ModelListGP(*objective_models)
