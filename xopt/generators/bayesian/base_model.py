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

    Attributes
    ----------
    name : str
        The name of the model.


    Methods
    -------
    build_model(input_names, outcome_names, data, input_bounds=None, dtype=torch.double, device='cpu')
        Build and return a trained botorch model for objectives and constraints.

    build_model_from_vocs(vocs, data, dtype=torch.double, device='cpu')
        Convenience wrapper around `build_model` for use with VOCs (Variables, Objectives,
        Constraints, Statics).

    build_single_task_gp(train_X, train_Y, **kwargs)
        Utility method for creating and training simple SingleTaskGP models.

    build_heteroskedastic_gp(train_X, train_Y, train_Yvar, **kwargs)
        Utility method for creating and training heteroskedastic SingleTaskGP models.

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
        """
        Build and return a trained botorch model for objectives and constraints.

        Parameters
        ----------
        input_names : List[str]
            Names of input variables.
        outcome_names : List[str]
            Names of outcome variables.
        data : pd.DataFrame
            Data used for training the model.
        input_bounds : Dict[str, List], optional
            Bounds for input variables.
        dtype : torch.dtype, optional
            Data type for the model (default is torch.double).
        device : Union[torch.device, str], optional
            Device on which to perform computations (default is "cpu").

        Returns
        -------
        ModelListGP
            The trained botorch model.

        """
        pass

    def build_model_from_vocs(
        self,
        vocs: VOCS,
        data: pd.DataFrame,
        dtype: torch.dtype = torch.double,
        device: Union[torch.device, str] = "cpu",
    ):
        """
        Convenience wrapper around `build_model` for use with VOCS (Variables,
        Objectives, Constraints, Statics).

        Parameters
        ----------
        vocs : VOCS
            The VOCS object for defining the problem's variables, objectives,
            constraints, and statics.
        data : pd.DataFrame
            Data used for training the model.
        dtype : torch.dtype, optional
            Data type for the model (default is torch.double).
        device : Union[torch.device, str], optional
            Device on which to perform computations (default is "cpu").

        Returns
        -------
        ModelListGP
            The trained botorch model.

        """
        return self.build_model(
            vocs.variable_names, vocs.output_names, data, vocs.variables, dtype, device
        )

    @staticmethod
    def build_single_task_gp(train_X: Tensor, train_Y: Tensor, **kwargs) -> Model:
        """
        Utility method for creating and training simple SingleTaskGP models.

        Parameters
        ----------
        train_X : Tensor
            Training data for input variables.
        train_Y : Tensor
            Training data for outcome variables.
        **kwargs
            Additional keyword arguments for model configuration.

        Returns
        -------
        Model
            The trained SingleTaskGP model.

        """
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
        """
        Utility method for creating and training heteroskedastic SingleTaskGP models.

        Parameters
        ----------
        train_X : Tensor
            Training data for input variables.
        train_Y : Tensor
            Training data for outcome variables.
        train_Yvar : Tensor
            Training data for outcome variable variances.
        **kwargs
            Additional keyword arguments for model configuration.

        Returns
        -------
        Model
            The trained heteroskedastic SingleTaskGP model.

        Notes
        -----
        Heteroskedastic modeling incurs a number of warnings from botorch, which are
        suppressed within this method.

        """
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
