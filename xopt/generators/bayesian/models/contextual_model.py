import numpy as np
import pandas as pd
import torch
from botorch.models import ModelListGP
from botorch.models.transforms import Normalize

from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.bayesian.options import ModelOptions
from xopt.vocs import VOCS


class ContextualModelOptions(ModelOptions):
    name: str = "contextual_standard"


class ContextualModelConstructor(StandardModelConstructor):
    def __init__(self, vocs: VOCS, options: ContextualModelOptions):
        if not type(options) is ContextualModelOptions:
            raise ValueError("options must be a ContextualModelOptions object")
        super().__init__(vocs, options)

    def build_model(self, data: pd.DataFrame, tkwargs: dict = None) -> ModelListGP:
        self.tkwargs = tkwargs or {"dtype": torch.double, "device": "cpu"}
        self.collect_data(data)

        # add time column to variable data
        self.input_data = pd.concat([self.input_data, data["context"]], axis=1)

        # add bounds for input transformation
        bounds = np.hstack(
            [
                self.vocs.bounds,
                np.array(
                    (
                        data["context"].to_numpy().min(),
                        data["context"].to_numpy().max(),
                    )
                ).reshape(2, -1),
            ]
        )

        self.input_transform = Normalize(
            self.vocs.n_variables + 1, bounds=torch.tensor(bounds, **tkwargs)
        )
        self.input_transform.to(**tkwargs)

        return self.build_standard_model()
