from copy import deepcopy
from typing import Union

import pandas as pd

import torch

from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.vocs import VOCS


class TimeDependentModelConstructor(StandardModelConstructor):
    name = "time_dependent"

    def build_model_from_vocs(
        self,
        vocs: VOCS,
        data: pd.DataFrame,
        dtype: torch.dtype = torch.double,
        device: Union[torch.device, str] = "cpu",
    ):
        # get max/min times
        min_t = data["time"].min()
        max_t = data["time"].max() + 15.0
        variable_dict = deepcopy(vocs.variables)
        variable_dict["time"] = [min_t, max_t]
        return self.build_model(
            vocs.variable_names + ["time"],
            vocs.output_names,
            data,
            variable_dict,
            dtype,
            device,
        )
