import botorch
import torch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from ..utils import get_bounds, standardize


def create_model(train_x, train_y, train_c, vocs, custom_model=None, **kwargs):
    # create model
    if custom_model is None:
        # standardize y training data - use xopt version to allow for nans
        standardized_train_y = standardize(train_y)

        # horiz. stack objective and constraint results for training/acq specification
        train_outputs = torch.hstack((standardized_train_y, train_c))

        input_normalize = Normalize(len(vocs['variables']), get_bounds(vocs))

        model = SingleTaskGP(train_x, train_outputs,
                             input_transform=input_normalize, **kwargs)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

    else:
        model = custom_model(train_x, train_y, train_c, vocs, **kwargs)
        assert isinstance(model, Model)

    return model


def create_multi_fidelity_model(train_x, train_obj, train_c, vocs):
    assert list(vocs['variables'])[-1] == 'cost', 'last variable in vocs["variables"] must be "cost"'
    input_normalize = Normalize(len(vocs['variables']), get_bounds(vocs, device=train_x.device))
    model = SingleTaskMultiFidelityGP(
        train_x,
        train_obj,
        #input_transform=input_normalize,
        outcome_transform=Standardize(m=1),
        data_fidelity=len(vocs['variables']) - 1
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


def create_simple_multi_fidelity_model(train_x, train_obj, train_c, vocs):
    assert list(vocs['variables'])[-1] == 'cost', 'last variable in vocs["variables"] must be "cost"'
    input_normalize = Normalize(len(vocs['variables']), get_bounds(vocs, device=train_x.device))
    model = SingleTaskMultiFidelityGP(
        train_x,
        train_obj,
        #input_transform=input_normalize,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model
