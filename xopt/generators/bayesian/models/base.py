from abc import ABC, abstractmethod

import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.model import Model
from gpytorch import ExactMarginalLogLikelihood
from torch import Tensor

from xopt.vocs import VOCS


class ModelConstructor(ABC):
    """
    Abstract class that defines instructions for building heterogeneous botorch models
    used in Xopt Bayesian generators.

    """

    def __init__(self, vocs: VOCS, tkwargs: dict = None):
        self.vocs = vocs
        self.tkwargs = tkwargs or {"dtype": torch.double, "device": "cpu"}

    @abstractmethod
    def build_model(self, data: pd.DataFrame) -> ModelListGP:
        """return a trained botorch model for objectives and constraints"""
        pass

    @abstractmethod
    def build_objective_model(
        self, train_X: Tensor, train_Y: Tensor
    ) -> Model:
        pass

    @abstractmethod
    def build_constraint_model(
        self, train_X: Tensor, train_Y: Tensor
    ) -> Model:
        pass

    @staticmethod
    def _build_single_task_gp(train_X: Tensor, train_Y: Tensor, **kwargs) -> Model:
        model = SingleTaskGP(train_X, train_Y, **kwargs)

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model
