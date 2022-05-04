from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd
import torch
from botorch import fit_gpytorch_model
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch import Module
from gpytorch.mlls import ExactMarginalLogLikelihood

from xopt.generator import Generator
from xopt.generators.bayesian.custom_acq.proximal import ProximalAcquisitionFunction
from xopt.generators.bayesian.options import BayesianOptions
from xopt.vocs import VOCS


class BayesianGenerator(Generator, ABC):
    def __init__(self, vocs: VOCS, options: BayesianOptions = BayesianOptions()):
        if not isinstance(options, BayesianOptions):
            raise ValueError("options must be of type BayesianOptions")

        super(BayesianGenerator, self).__init__(vocs, options)

        self._model = None
        self._acquisition = None
        self._sampler = None
        self._objective = self._get_objective()

        # as a default normalize the inputs for the GP model
        bounds = torch.tensor(vocs.bounds)
        self._input_transform = Normalize(
            len(bounds[0]), bounds=bounds
        )

        # as a default standardize the outcomes for the GP model
        self._outcome_transform = Standardize(1)

        self._tkwargs = {"dtype": torch.double, "device": "cpu"}

    def generate(self, n_candidates: int) -> List[Dict]:

        # if no data exists use random generator to generate candidates
        if self.data.empty:
            return self.vocs.random_inputs(self.options.n_initial)

        else:
            bounds = torch.tensor(self.vocs.bounds, **self._tkwargs)

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
        return torch.tensor(inputs, **self._tkwargs), torch.tensor(
            outputs, **self._tkwargs
        )

    def get_acquisition(self, model):
        """
        Returns a function that can be used to evaluate the acquisition function
        """
        # re-create sampler/objective from options
        self._sampler = SobolQMCNormalSampler(self.options.acq.monte_carlo_samples)
        self._objective = self._get_objective()

        # add proximal biasing if requested
        if self.options.acq.proximal_lengthscales is not None:
            acq = ProximalAcquisitionFunction(
                self._get_acquisition(model),
                torch.tensor(
                    self.options.acq.proximal_lengthscales, **self._tkwargs
                ),
            )
        else:
            acq = self._get_acquisition(model)

        return acq

    @abstractmethod
    def _get_acquisition(self, model):
        pass

    @abstractmethod
    def _get_objective(self):
        pass

    @property
    def model(self):
        return self._model
