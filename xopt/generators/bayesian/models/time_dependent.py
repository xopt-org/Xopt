import numpy as np
import pandas as pd
import torch
from botorch.models import ModelListGP
from botorch.models.transforms import Normalize

from xopt.generators.bayesian.models.standard import (
    create_constraint_models,
    create_objective_models,
)
from xopt.generators.bayesian.models.utils import split_data


def create_time_dependent_model(
    data,
    vocs,
    added_time: float = 0.0,
    use_conservative_prior_lengthscale: bool = False,
    use_conservative_prior_mean: bool = False,
    use_low_noise_prior: bool = False,
) -> ModelListGP:

    # create dataframes for processed data
    input_data, objective_data, constraint_data = split_data(data, vocs)
    # add time column to variable data
    input_data = pd.concat([input_data, data["time"]], axis=1)
    # add bounds for input transformation
    bounds = np.hstack(
        [
            vocs.bounds,
            np.array(
                (
                    data["time"].to_numpy().min(),
                    data["time"].to_numpy().max() + 2 * added_time,
                )
            ).reshape(2, 1),
        ]
    )

    tkwargs = {"dtype": torch.double, "device": "cpu"}
    input_transform = Normalize(
        vocs.n_variables + 1, bounds=torch.tensor(bounds, **tkwargs)
    )

    objective_models = create_objective_models(
        input_data, objective_data, input_transform, tkwargs, use_low_noise_prior
    )
    constraint_models = create_constraint_models(
        input_data,
        constraint_data,
        input_transform,
        tkwargs,
        use_low_noise_prior,
        use_conservative_prior_lengthscale,
        use_conservative_prior_mean,
    )

    return ModelListGP(*objective_models, *constraint_models)
