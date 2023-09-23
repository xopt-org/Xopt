from abc import ABC, abstractmethod
from typing import Dict, List, Union

import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.model import Model
from gpytorch import ExactMarginalLogLikelihood
from pydantic import ConfigDict
from torch import Tensor

from xopt.generators.bayesian.custom_botorch.heteroskedastic import (
    XoptHeteroskedasticSingleTaskGP,
)
from xopt.pydantic import XoptBaseModel
from xopt.vocs import VOCS


class ModelConstructor(XoptBaseModel, ABC):
    """
    Abstract class that defines instructions for building heterogeneous botorch models
    used in Xopt Bayesian generators.

    """

    name: str

    model_config = ConfigDict(validate_assignment=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def build_model(
        self,
        input_names: List[str],
        outcome_names: List[str],
        data: pd.DataFrame,
        input_bounds: Dict[str, List] = None,
        dtype: torch.dtype = torch.double,
        device: Union[torch.device, str] = "cpu",
    ) -> ModelListGP:
        """return a trained botorch model for objectives and constraints"""
        pass

    def build_model_from_vocs(
        self,
        vocs: VOCS,
        data: pd.DataFrame,
        dtype: torch.dtype = torch.double,
        device: Union[torch.device, str] = "cpu",
    ):
        """convenience wrapper around build model for use in xopt w vocs"""
        return self.build_model(
            vocs.variable_names, vocs.output_names, data, vocs.variables, dtype, device
        )

    @staticmethod
    def build_single_task_gp(train_X: Tensor, train_Y: Tensor, **kwargs) -> Model:
        """utility method for creating and training simple SingleTaskGP models"""
        if train_X.shape[0] == 0 or train_Y.shape[0] == 0:
            raise ValueError("no data found to train model!")
        model = SingleTaskGP(train_X, train_Y, **kwargs)

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    @staticmethod
    def build_heteroskedastic_gp(
        train_X: Tensor, train_Y: Tensor, train_Yvar: Tensor, **kwargs
    ) -> Model:
        """utility method for creating and
        training heteroskedastic SingleTaskGP models"""
        # heteroskedastic modeling incurs a number of warnings from botorch,
        # we suppress those here.
        import warnings

        warnings.filterwarnings("ignore")

        if train_X.shape[0] == 0 or train_Y.shape[0] == 0:
            raise ValueError("no data found to train model!")
        model = XoptHeteroskedasticSingleTaskGP(
            train_X,
            train_Y,
            train_Yvar,
            **kwargs,
        )

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model
