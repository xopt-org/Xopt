import numpy as np
import pandas as pd
from botorch.models import ModelListGP

from xopt.generators.bayesian.models.standard import create_split_model
from xopt.generators.bayesian.models.utils import split_data


def create_time_dependent_model(data, vocs, added_time=0, **kwargs) -> ModelListGP:

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

    return create_split_model(
        input_data, objective_data, constraint_data, bounds, **kwargs
    )
