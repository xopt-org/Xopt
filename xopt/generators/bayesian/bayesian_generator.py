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
from .custom_acq.proximal import ProximalAcquisitionFunction
from .objectives import create_constrained_mc_objective
from .options import BayesianOptions


class BayesianGenerator(Generator, ABC):
    def __init__(
        self, vocs: VOCS, options: BayesianOptions = BayesianOptions(), **kwargs
    ):
        super(BayesianGenerator, self).__init__(vocs)

        self._model = None
        self._acquisition = None

        self.options = options

        if self.options.acq.objective is None:
            self.options.acq.objective = create_constrained_mc_objective(self.vocs)

        if self.options.model.input_transform is None:
            bounds = torch.tensor(vocs.bounds, **self.options.tkwargs)
            self.options.model.input_transform = Normalize(
                len(bounds[0]), bounds=bounds
            )

        if self.options.model.outcome_transform is None:
            self.options.model.outcome_transform = Standardize(1)

        # update generator options w/kwargs
        self.options.update(**kwargs)

    def generate(self, n_candidates: int) -> List[Dict]:

        # if no data exists use random generator to generate candidates
        if self.data.empty:
            return self.vocs.random_inputs(self.options.n_initial)

        else:
            bounds = torch.tensor(self.vocs.bounds, **self.options.tkwargs)

            # update internal model with internal data
            self.train_model(self.data)

            candidates, out = optimize_acqf(
                acq_function=self.get_acquisition(self._model),
                bounds=bounds,
                q=n_candidates,
                **self.options.optim.dict()
            )
            return self.vocs.convert_numpy_to_inputs(candidates.detach().numpy())

    def train_model(self, data: pd.DataFrame = None, update_internal=True) -> Module:
        """
        Returns a SingleTaskGP (or ModelList set of independent SingleTaskGPs
        depending on the number of outputs, if data is None

        """

        inputs, outputs = self.get_training_data(data)

        n_outputs = outputs.shape[-1]
        if n_outputs > 1:
            model_list = []
            for i in range(n_outputs):
                m = SingleTaskGP(
                    inputs, outputs[:, i].unsqueeze(1), **self.options.model.dict()
                )
                mll = ExactMarginalLogLikelihood(m.likelihood, m)
                fit_gpytorch_model(mll)
                model_list += [m]

            model = ModelListGP(*model_list)
        else:
            model = SingleTaskGP(inputs, outputs, **self.options.model.dict())

            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

        if update_internal:
            self._model = model
        return model

    def get_training_data(self, data: pd.DataFrame) -> (torch.Tensor, torch.Tensor):
        """overwrite get training data to transform numpy array into tensor"""
        inputs, outputs = self.vocs.get_training_data(data)
        return torch.tensor(inputs, **self.options.tkwargs), torch.tensor(
            outputs, **self.options.tkwargs
        )

    def get_acquisition(self, model):
        """
        Returns a function that can be used to evaluate the acquisition function
        """
        # add proximal biasing if requested
        if self.options.proximal_lengthscales is not None:
            acq = ProximalAcquisitionFunction(
                self._get_acquisition(model),
                torch.tensor(
                    self.options.proximal_lengthscales, **self.options.tkwargs
                ),
            )
        else:
            acq = self._get_acquisition(model)

        return acq

    @abstractmethod
    def _get_acquisition(self, model):
        pass

    @property
    def model(self):
        return self._model
