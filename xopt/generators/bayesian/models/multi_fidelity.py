from typing import Dict

import numpy as np
import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP, SingleTaskMultiFidelityGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior
from pandas import DataFrame
from pydantic import Field

from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.bayesian.options import ModelOptions
from xopt.vocs import VOCS


class MultiFidelityModelOptions(ModelOptions):
    fidelity_parameter: str = Field("s", description="fidelity parameter name")


class MultiFidelityModelConstructor(StandardModelConstructor):
    def __init__(self, vocs: VOCS, options):
        if len(vocs.constraints) > 0:
            raise NotImplementedError(
                "Constraints are not allowed for multi-fidelity models yet"
            )
        if len(vocs.objectives) > 1:
            raise NotImplementedError(
                "Multiple objectives are not allowed for multi-fidelity models yet"
            )

        super(MultiFidelityModelConstructor, self).__init__(vocs, options)

    def collect_data(self, data: pd.DataFrame):
        super().collect_data(data)
        self.input_data = pd.concat(
            (self.input_data, data[self.options.fidelity_parameter].dropna()), axis=1
        )
        self.objective_data = pd.concat(
            (self.objective_data, data[self.options.fidelity_parameter].dropna()), axis=1
        )
        self.train_X = torch.tensor(self.input_data.to_numpy(), **self.tkwargs)

    def build_model(self, data: pd.DataFrame, tkwargs: dict = None) -> SingleTaskGP:
        """construct independent model for each objective and constraint"""
        # overwrite tkwargs if specified
        self.tkwargs = tkwargs or self.tkwargs

        # collect data from dataframe
        self.collect_data(data)

        # build model
        return self.build_multi_fidelity_model()

    def build_multi_fidelity_model(self):
        # augment normal bounds to get the multi-fidelity bounds
        mf_bounds = np.hstack((self.vocs.bounds, np.array([0, 1]).reshape(2, 1)))
        input_transform = Normalize(
            self.vocs.n_variables + 1, bounds=torch.tensor(mf_bounds, **self.tkwargs)
        )
        # check bounds on fidelity data
        if (max(self.train_X[:, -1]) > 1).any() or (min(self.train_X[:, -1]) < 0).any():
            raise ValueError("fidelity values must be in the domain [0,1]")

        # create models for objectives
        objective_name = self.vocs.objective_names[0]
        train_Y = torch.tensor(
            self.objective_data[
                [objective_name, self.options.fidelity_parameter]
            ].to_numpy(),
            **self.tkwargs
        )
        if len(train_Y.shape) == 1:
            train_Y = train_Y.unsqueeze(-1)

        model = SingleTaskGP(
            self.train_X,
            train_Y,
            input_transform=input_transform,
            outcome_transform=Standardize(m=train_Y.shape[-1]),
            likelihood=self.likelihood,
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        return model
