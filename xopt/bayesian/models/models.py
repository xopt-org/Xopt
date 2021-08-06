import botorch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch import ExactMarginalLogLikelihood


def create_model(train_x, train_outputs, input_normalize, custom_model=None, **kwargs):
    # create model
    if custom_model is None:
        model = SingleTaskGP(train_x, train_outputs,
                             input_transform=input_normalize, **kwargs)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

    else:
        model = custom_model(train_x, train_outputs,
                             input_transform=input_normalize, **kwargs)
        assert isinstance(model, botorch.models.model.Model)

    return model

