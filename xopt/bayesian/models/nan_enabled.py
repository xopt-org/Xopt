import torch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.model_list_gp_regression import ModelListGP


class ModelCreationError(Exception):
    pass


def get_nan_model(train_x, train_outputs, input_transform, outcome_transform):
    """
    Model that allows for Nans by splitting up each objective/constraint that has nans
    into seperate GP models using IndependentModelList.

    For each training data point that has a Nan in the output, that training data
    point is removed from the corresponding model.

    """

    combined_outputs = train_outputs
    n_outputs = combined_outputs.shape[-1]

    gp_models = []
    for ii in range(n_outputs):
        output = combined_outputs[:, ii].flatten()

        nan_state = torch.isnan(output)
        not_nan_idx = torch.nonzero(~nan_state).flatten()

        # remove elements that have nan values
        temp_train_x = train_x[not_nan_idx]
        temp_train_y = output[not_nan_idx].reshape(-1, 1)

        if len(temp_train_y) == 0:
            raise ModelCreationError("No valid measurements passed to model")

        # create single task model and add to list
        submodel = SingleTaskGP(
            temp_train_x,
            temp_train_y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )

        mll = ExactMarginalLogLikelihood(submodel.likelihood, submodel)
        fit_gpytorch_model(mll)

        gp_models.append(submodel)

    return ModelListGP(*gp_models)
