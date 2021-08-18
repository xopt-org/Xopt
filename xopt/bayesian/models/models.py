import botorch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize

from gpytorch import ExactMarginalLogLikelihood
import torch
from ..utils import standardize, get_bounds


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
        assert isinstance(model, botorch.models.model.Model)

    return model
