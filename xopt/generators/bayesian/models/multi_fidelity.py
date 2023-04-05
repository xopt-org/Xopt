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


def create_multifidelity_model(
    data: pd.DataFrame, vocs: VOCS, fidelity_key: str, use_low_noise_prior: bool = False
):
    _, objective_data, constraint_data = vocs.extract_data(data)
    input_data = data[vocs.variable_names + [fidelity_key]]

    if not constraint_data.empty:
        raise NotImplementedError(
            "Constraints are not allowed for multi-fidelity models"
        )

    tkwargs = {"dtype": torch.double, "device": "cpu"}

    # augment normal bounds to get the multi-fidelity bounds
    mf_bounds = np.hstack((vocs.bounds, np.array([0, 1]).reshape(2, 1)))
    input_transform = Normalize(
        vocs.n_variables + 1, bounds=torch.tensor(mf_bounds, **tkwargs)
    )

    objective_model = create_multi_fidelity_objective_models(
        input_data,
        objective_data,
        vocs.objective_names[0],
        input_transform,
        tkwargs,
        use_low_noise_prior,
    )

    return objective_model


def create_multi_fidelity_objective_models(
    input_data: DataFrame,
    objective_data: DataFrame,
    objective_name: str,
    input_transform: Normalize,
    tkwargs: Dict,
    use_low_noise_prior: bool = False,
):
    """
    Creates a multifidelity model using Botorch. Since botorch is aimed at optimizing NN
    hyperparameters we adapt it to our uses here by setting the iteration fidelity index
    to the simulation fidelity index (uses a Exponential Decay Kernel non-stationary
    kernel).
    """

    # validate data
    if len(input_data) == 0:
        raise ValueError("input_data is empty/all Nans, cannot create model")

    train_X = torch.tensor(input_data.to_numpy(), **tkwargs)

    # check bounds on fidelity data
    if (max(train_X[:, -1]) > 1).any() or (min(train_X[:, -1]) < 0).any():
        raise ValueError("fidelity values must be in the domain [0,1]")

    # get fidelity index
    f_index = -1

    # create models for objectives
    train_Y = torch.tensor(
        objective_data[objective_name].to_numpy(), **tkwargs
    ).unsqueeze(-1)

    likelihood = None
    if use_low_noise_prior:
        likelihood = GaussianLikelihood(noise_prior=GammaPrior(1.0, 10.0))

    model = SingleTaskMultiFidelityGP(
        train_X,
        train_Y,
        input_transform=input_transform,
        outcome_transform=Standardize(1),
        likelihood=likelihood,
        iteration_fidelity=f_index,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    return model
