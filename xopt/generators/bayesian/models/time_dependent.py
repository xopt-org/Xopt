import numpy as np
import pandas as pd
import torch
from botorch.models.transforms import Normalize

from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.vocs import VOCS


class TimeDependentModelConstructor(StandardModelConstructor):
    name = "time_dependent"

    def get_input_transform(self, vocs: VOCS, data: pd.DataFrame):
        # add bounds for input transformation
        bounds = np.hstack(
            [
                vocs.bounds,
                np.array(
                    (
                        data["time"].to_numpy().min(),
                        data["time"].to_numpy().max() + 2 * 15.0,
                    )
                ).reshape(2, 1),
            ]
        )

        return Normalize(vocs.n_variables + 1, bounds=torch.tensor(bounds)).to(
            **self.tkwargs
        )

    def _get_training_data(
        self, name, vocs: VOCS, data
    ) -> (torch.Tensor, torch.Tensor):
        train_X, train_Y = super()._get_training_data(name, vocs, data)

        # append time data to last X axis
        time_X = torch.tensor(data["time"].to_numpy(), **self.tkwargs).unsqueeze(1)
        train_X = torch.cat((train_X, time_X), dim=-1)
        return train_X, train_Y
