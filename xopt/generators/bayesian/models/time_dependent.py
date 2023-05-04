import numpy as np
import pandas as pd
import torch
from botorch.models import ModelListGP
from botorch.models.transforms import Normalize
from pydantic import Field

from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.bayesian.options import ModelOptions
from xopt.vocs import VOCS


class TimeDependentModelOptions(ModelOptions):
    name: str = "time_dependent_standard"
    added_time: float = Field(
        1.0,
        description="additional time added to target time "
        "for optimization, make sure its "
        "larger than computation time for the "
        "GP model",
    )


class TimeDependentModelConstructor(StandardModelConstructor):
    def __init__(self, vocs: VOCS, options: TimeDependentModelOptions):
        if not type(options) is TimeDependentModelOptions:
            raise ValueError("options must be a TimeDependentModelOptions object")

        super().__init__(vocs, options)

    def build_model(self, data: pd.DataFrame, tkwargs: dict = None) -> ModelListGP:
        self.tkwargs = tkwargs or {"dtype": torch.double, "device": "cpu"}

        self.collect_data(data)

        # add time column to variable data
        self.input_data = pd.concat([self.input_data, data["time"]], axis=1)

        # add bounds for input transformation
        bounds = np.hstack(
            [
                self.vocs.bounds,
                np.array(
                    (
                        data["time"].to_numpy().min(),
                        data["time"].to_numpy().max() + 2 * self.options.added_time,
                    )
                ).reshape(2, 1),
            ]
        )

        self.input_transform = Normalize(
            self.vocs.n_variables + 1, bounds=torch.tensor(bounds, **tkwargs)
        )
        self.input_transform.to(**tkwargs)

        return self.build_standard_model()
