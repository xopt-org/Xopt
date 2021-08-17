import botorch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch import ExactMarginalLogLikelihood
import torch
from ..utils import standardize


def create_model(train_x, train_y, train_c, input_normalize, custom_model=None, **kwargs):
    # create model
    if custom_model is None:
        # standardize y training data - use xopt version to allow for nans
        standardized_train_y = standardize(train_y)

        # horiz. stack objective and constraint results for training/acq specification
        train_outputs = torch.hstack((standardized_train_y, train_c))

        model = SingleTaskGP(train_x, train_outputs,
                             input_transform=input_normalize, **kwargs)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

    else:
        model = custom_model(train_x, train_y, train_c,
                             input_transform=input_normalize, **kwargs)
        assert isinstance(model, botorch.models.model.Model)

    return model

