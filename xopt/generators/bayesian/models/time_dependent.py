from copy import deepcopy
from typing import Dict, List, Union

import pandas as pd
import torch
from botorch.models import ModelListGP
from gpytorch.kernels import (
    ProductKernel,
    SpectralMixtureKernel,
    MaternKernel,
)
from gpytorch.priors import GammaPrior
from pydantic import Field

from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.vocs import VOCS


class TimeDependentModelConstructor(StandardModelConstructor):
    name: str = Field("time_dependent", frozen=True)
    use_spectral_mixture_kernel: bool = True

    def build_model(
        self,
        input_names: List[str],
        outcome_names: List[str],
        data: pd.DataFrame,
        input_bounds: Dict[str, List] = None,
        dtype: torch.dtype = torch.double,
        device: Union[torch.device, str] = "cpu",
    ) -> ModelListGP:
        # get model input names
        new_input_names = deepcopy(input_names)
        new_input_names += ["time"]

        min_t = data["time"].min()
        max_t = data["time"].max() + 15.0
        new_input_bounds = deepcopy(input_bounds)
        new_input_bounds["time"] = [min_t, max_t]

        # set covar modules if not specified -- use SpectralMixtureKernel for time axis
        if len(self.covar_modules) == 0 and self.use_spectral_mixture_kernel:
            covar_modules = {}
            for name in outcome_names:
                if len(input_names) == 1:
                    matern_dims = [0]
                else:
                    matern_dims = tuple(range(len(input_names)))
                time_dim = [len(input_names)]

                matern_kernel = MaternKernel(
                    nu=2.5,
                    active_dims=matern_dims,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                )

                covar_modules[name] = ProductKernel(
                    SpectralMixtureKernel(num_mixtures=5, active_dims=time_dim),
                    matern_kernel,
                )
                self.covar_modules = covar_modules

        return super().build_model(
            new_input_names, outcome_names, data, new_input_bounds, dtype, device
        )

    def build_model_from_vocs(
        self,
        vocs: VOCS,
        data: pd.DataFrame,
        dtype: torch.dtype = torch.double,
        device: Union[torch.device, str] = "cpu",
    ):
        return self.build_model(
            vocs.variable_names + ["time"],
            vocs.output_names,
            data,
            vocs.variables,
            dtype,
            device,
        )
