import torch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.model import Model
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from ..models.nan_enabled import get_nan_model
from ..outcome_transforms import NanEnabledStandardize
from ...vocs_tools import get_bounds


def create_model(train_x, train_y, train_c, vocs, custom_model=None, **kwargs):
    input_normalize = Normalize(
        len(vocs.variables),
        bounds=torch.tensor(
            get_bounds(vocs),
        ).to(train_x),
    )

    # create model
    if custom_model is None:
        if train_c is not None:
            train_outputs = torch.hstack((train_y, train_c))
        else:
            train_outputs = train_y

        # test if there are nans in the training data
        if torch.any(torch.isnan(train_outputs)):
            output_standardize = NanEnabledStandardize(m=1)
            model = get_nan_model(
                train_x, train_outputs, input_normalize, output_standardize
            )
        else:
            output_standardize = Standardize(m=train_outputs.shape[-1])
            model = SingleTaskGP(
                train_x,
                train_outputs,
                input_transform=input_normalize,
                outcome_transform=output_standardize,
            )

            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

    else:
        model = custom_model(train_x, train_y, train_c, vocs, **kwargs)
        assert isinstance(model, Model)

    return model


def create_multi_fidelity_model(train_x, train_y, train_c, vocs):
    assert list(vocs.variables)[-1] == "cost", (
        'last variable in vocs.variables ' 'must be "cost" '
    )

    input_transform = Normalize(
        len(vocs.variables), torch.tensor(get_bounds(vocs)).to(train_x)
    )
    outcome_transform = Standardize(m=1)

    model = SingleTaskMultiFidelityGP(
        train_x,
        train_y,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
        linear_truncated=False,
        iteration_fidelity=len(vocs.variables) - 1,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model
