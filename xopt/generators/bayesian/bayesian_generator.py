from abc import ABC, abstractmethod
from typing import List, Dict

import pandas as pd
import torch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP, ModelListGP, ModelList
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from gpytorch import Module
from gpytorch.mlls import ExactMarginalLogLikelihood

from xopt import Generator, VOCS
from xopt.generators.random import RandomGenerator
from .utils import create_constrained_mc_objective


class BayesianGenerator(Generator, ABC):
    _model = None
    _acquisition = None

    def __init__(
        self,
        vocs: VOCS,
        n_initial: int = 1,
        model_kw: Dict = None,
        acqf_kw: Dict = None,
        optim_kw: Dict = None,
    ):
        super(BayesianGenerator, self).__init__(vocs)
        self.n_initial = n_initial
        self.tkwargs = {"dtype": torch.double, "device": "cpu"}

        # set default kwargs
        model_kw = model_kw or {}
        acqf_kw = acqf_kw or {}
        optim_kw = optim_kw or {}

        # acquisition function
        self.acqf_kw = {"objective": create_constrained_mc_objective(self.vocs)}

        self.acqf_kw.update(acqf_kw)

        # kwargs for optimizing acquisition function
        self.optim_kw = {"num_restarts": 5, "raw_samples": 20}
        self.optim_kw.update(optim_kw)

        # kwargs for specifying model construction
        bounds = self.get_bounds()
        self.model_kw = {
            "input_transform": Normalize(len(bounds[0]), bounds=bounds),
            "outcome_transform": Standardize(1),
        }
        self.model_kw.update(model_kw)

    def generate(self, data: pd.DataFrame, n_candidates) -> List[Dict]:

        # if no data exists use random generator to generate candidates
        if data.empty:
            gen = RandomGenerator(self.vocs)
            return gen.generate(data, self.n_initial)

        else:
            self._model = self.get_model(data)
            self._acquisition = self.get_acquisition(self.model)
            bounds = self.get_bounds()
            candidates, _ = optimize_acqf(
                acq_function=self.acquisition,
                bounds=bounds,
                q=n_candidates,
                **self.optim_kw
            )
            return self.convert_numpy_candidates(candidates.detach().numpy())

    def get_model(self, data: pd.DataFrame) -> Module:
        """
        Returns a SingleTaskGP (or ModelList set of independent SingleTaskGPs
        depending on the number of outputs

        """
        inputs, outputs = self.get_training_data(data)

        n_outputs = outputs.shape[-1]
        if n_outputs > 1:
            model_list = []
            for i in range(n_outputs):
                m = SingleTaskGP(inputs, outputs[:, i].unsqueeze(1), **self.model_kw)
                mll = ExactMarginalLogLikelihood(m.likelihood, m)
                fit_gpytorch_model(mll)
                model_list += [m]

            model = ModelListGP(*model_list)
        else:
            model = SingleTaskGP(inputs, outputs, **self.model_kw)

            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)
        return model

    def get_bounds(self):
        """overwrite get bounds to transform numpy array into tensor"""
        return torch.tensor(super().get_bounds(), **self.tkwargs)

    def get_training_data(self, data: pd.DataFrame):
        """overwrite get training data to transform numpy array into tensor"""
        inputs, outputs = super().get_training_data(data)
        return torch.tensor(inputs, **self.tkwargs), torch.tensor(
            outputs, **self.tkwargs
        )

    @abstractmethod
    def get_acquisition(self, model):
        pass

    @property
    def model(self):
        return self._model

    @property
    def acquisition(self):
        return self._acquisition
