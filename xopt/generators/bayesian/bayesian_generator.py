from abc import ABC, abstractmethod
from typing import List, Dict

import pandas as pd
import torch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP, ModelListGP
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
            vocs: VOCS
    ):
        super(BayesianGenerator, self).__init__(vocs)
        tkwargs = {"dtype": torch.double, "device": "cpu"}
        self.options.update({"tkwargs": tkwargs})

        # kwargs for acquisition function
        acqf_kw = {"objective": create_constrained_mc_objective(self.vocs)}

        # kwargs for optimizing the acquisition function
        optim_kw = {"num_restarts": 5, "raw_samples": 20}

        # kwargs for specifying model construction
        bounds = self.get_bounds()
        model_kw = {
            "input_transform": Normalize(len(bounds[0]), bounds=bounds),
            "outcome_transform": Standardize(1),
        }

        # add default optional arguments
        self.options.update({"n_initial": 1})
        self.options.update({"optim_kw": optim_kw})
        self.options.update({"acqf_kw": acqf_kw})
        self.options.update({"model_kw": model_kw})

    def generate(self, n_candidates) -> List[Dict]:

        # if no data exists use random generator to generate candidates
        if self.data.empty:
            gen = RandomGenerator(self.vocs)
            return gen.generate(self.options["n_initial"])

        else:
            self._model = self.get_model()
            self._acquisition = self.get_acquisition(self.model)
            bounds = self.get_bounds()
            candidates, _ = optimize_acqf(
                acq_function=self.acquisition,
                bounds=bounds,
                q=n_candidates,
                **self.options["optim_kw"]
            )
            return self.convert_numpy_candidates(candidates.detach().numpy())

    def get_model(self, data: pd.DataFrame = None) -> Module:
        """
        Returns a SingleTaskGP (or ModelList set of independent SingleTaskGPs
        depending on the number of outputs

        """
        inputs, outputs = self.get_training_data(data)

        n_outputs = outputs.shape[-1]
        if n_outputs > 1:
            model_list = []
            for i in range(n_outputs):
                m = SingleTaskGP(
                    inputs, outputs[:, i].unsqueeze(1), **self.options["model_kw"]
                )
                mll = ExactMarginalLogLikelihood(m.likelihood, m)
                fit_gpytorch_model(mll)
                model_list += [m]

            model = ModelListGP(*model_list)
        else:
            model = SingleTaskGP(inputs, outputs, **self.options["model_kw"])

            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)
        return model

    def get_bounds(self):
        """overwrite get bounds to transform numpy array into tensor"""
        return torch.tensor(super().get_bounds(), **self.options["tkwargs"])

    def get_training_data(self, data: pd.DataFrame = None):
        """overwrite get training data to transform numpy array into tensor"""
        inputs, outputs = super().get_training_data(data)
        return torch.tensor(inputs, **self.options["tkwargs"]), torch.tensor(
            outputs, **self.options["tkwargs"]
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
