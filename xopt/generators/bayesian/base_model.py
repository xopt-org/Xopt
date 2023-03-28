from abc import ABC, abstractmethod

import pandas as pd
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

    def __init__(self, vocs: VOCS, options):
        self.vocs = vocs
        self.options = options

    @abstractmethod
    def build_model(self, data: pd.DataFrame, tkwargs: dict = None) -> ModelListGP:
        """return a trained botorch model for objectives and constraints"""
        pass

    @staticmethod
    def build_single_task_gp(train_X: Tensor, train_Y: Tensor, **kwargs) -> Model:
        """convience method for creating and training simple SingleTaskGP models"""
        if train_X.shape[0] == 0 or train_Y.shape[0] == 0:
            raise ValueError("no data found to train model!")
        model = SingleTaskGP(train_X, train_Y, **kwargs)

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model
