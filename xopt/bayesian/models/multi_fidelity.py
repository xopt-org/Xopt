from botorch import fit_gpytorch_model
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


def generate_multi_fidelity_model(train_x, train_obj, train_c, input_transform=None):
    model = SingleTaskMultiFidelityGP(
        train_x,
        train_obj,
        input_transform=input_transform,
        outcome_transform=Standardize(m=1),
        data_fidelity=6
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model
