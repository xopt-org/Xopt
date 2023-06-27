from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd
from botorch import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.model import Model
from gpytorch import ExactMarginalLogLikelihood
from torch import Tensor

from xopt.pydantic import XoptBaseModel
from xopt.vocs import VOCS


class ModelConstructor(XoptBaseModel, ABC):
    """
    Abstract class that defines instructions for building heterogeneous botorch models
    used in Xopt Bayesian generators.

    """

    name: str = None

    class Config:
        validate_assignment = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def build_model(
        self,
        input_names: List[str],
        outcome_names: List[str],
        data: pd.DataFrame,
        input_bounds: Dict[str, List] = None,
    ) -> ModelListGP:
        """return a trained botorch model for objectives and constraints"""
        pass

    def build_model_from_vocs(self, vocs: VOCS, data: pd.DataFrame):
        """convience wrapper around build model for use in xopt w vocs"""
        return self.build_model(
            vocs.variable_names, vocs.output_names, data, vocs.variables
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
