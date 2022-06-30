from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd
import torch
from botorch.acquisition import ProximalAcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.optim.initializers import sample_truncated_normal_perturbations
from botorch.sampling import SobolQMCNormalSampler
from gpytorch import Module

from xopt.generator import Generator
from xopt.generators.bayesian.models.standard import create_standard_model
from xopt.generators.bayesian.options import BayesianOptions
from xopt.vocs import VOCS


class BayesianGenerator(Generator, ABC):
    def __init__(self, vocs: VOCS, options: BayesianOptions = BayesianOptions()):
        if not isinstance(options, BayesianOptions):
            raise ValueError("options must be of type BayesianOptions")

        super(BayesianGenerator, self).__init__(vocs, options)

        self._model = None
        self._acquisition = None
        self.sampler = SobolQMCNormalSampler(self.options.acq.monte_carlo_samples)
        self.objective = self._get_objective()

        self._tkwargs = {"dtype": torch.double, "device": "cpu"}

    def add_data(self, new_data: pd.DataFrame):
        self.data = pd.concat([self.data, new_data], axis=0)

    def generate(self, n_candidates: int) -> List[Dict]:

        if n_candidates > 1:
            raise NotImplementedError(
                "Bayesian algorithms don't support parallel " "candidate generation"
            )

        # if no data exists use random generator to generate candidates
        if self.data.empty:
            return self.vocs.random_inputs(self.options.n_initial)

        else:
            bounds = torch.tensor(self.vocs.bounds, **self._tkwargs)

            # update internal model with internal data
            self.train_model(self.data)

            # generate starting points for optimization (note in real domain)
            inputs = self.get_input_data(self.data)
            batch_initial_points = sample_truncated_normal_perturbations(
                inputs[-1].unsqueeze(0),
                n_discrete_points=self.options.optim.raw_samples,
                sigma=0.5,
                bounds=bounds,
            )

            # get candidates in real domain
            candidates, out = optimize_acqf(
                acq_function=self.get_acquisition(self._model),
                bounds=bounds,
                q=n_candidates,
                batch_initial_conditions=batch_initial_points,
                num_restarts=self.options.optim.num_restarts,
            )
            return self.vocs.convert_numpy_to_inputs(
                candidates.unsqueeze(0).detach().numpy()
            )

    def train_model(self, data: pd.DataFrame, update_internal=True) -> Module:
        """
        Returns a ModelListGP containing independent models for the objectives and
        constraints

        """

        _model = create_standard_model(
            data, self.vocs, tkwargs=self._tkwargs, **self.options.model.dict()
        )

        if update_internal:
            self._model = _model
        return _model

    def get_acquisition(self, model):
        """
        Returns a function that can be used to evaluate the acquisition function
        """
        # re-create sampler/objective from options
        self.sampler = SobolQMCNormalSampler(self.options.acq.monte_carlo_samples)
        self.objective = self._get_objective()

        # add proximal biasing if requested
        if self.options.acq.proximal_lengthscales is not None:
            acq = ProximalAcquisitionFunction(
                self._get_acquisition(model),
                torch.tensor(self.options.acq.proximal_lengthscales, **self._tkwargs),
            )
        else:
            acq = self._get_acquisition(model)

        return acq

    def get_training_data(self, data: pd.DataFrame):
        return self.get_input_data(data), self.get_outcome_data(data)

    def get_input_data(self, data: pd.DataFrame):
        return torch.tensor(data[self.vocs.variable_names].to_numpy(), **self._tkwargs)

    def get_outcome_data(self, data: pd.DataFrame):
        return torch.tensor(data[self.vocs.output_names].to_numpy(), **self._tkwargs)

    @abstractmethod
    def _get_acquisition(self, model):
        pass

    @abstractmethod
    def _get_objective(self):
        pass

    @property
    def model(self):
        return self._model

    def _check_options(self, options: BayesianOptions):
        if options.acq.proximal_lengthscales is not None:
            n_lengthscales = len(options.acq.proximal_lengthscales)

            if n_lengthscales != self.vocs.n_variables:
                raise ValueError(
                    f"Number of proximal lengthscales ({n_lengthscales}) must match "
                    f"number of variables {self.vocs.n_variables}"
                )
            if options.optim.num_restarts != 1:
                raise ValueError(
                    "`options.optim.num_restarts` must be 1 when proximal biasing is "
                    "specified"
                )
