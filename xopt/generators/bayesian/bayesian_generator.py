from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd
import torch
from botorch import fit_gpytorch_model
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.optim.initializers import sample_truncated_normal_perturbations
from botorch.sampling import SobolQMCNormalSampler
from gpytorch import Module
from gpytorch.mlls import ExactMarginalLogLikelihood

from xopt.generator import Generator
from xopt.generators.bayesian.options import BayesianOptions
from xopt.vocs import VOCS
from xopt.generators.bayesian.custom_botorch.proximal import ProximalAcquisitionFunction


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
            raise NotImplementedError("Bayesian algorithms don't support parallel "
                                      "candidate generation")

        # if no data exists use random generator to generate candidates
        if self.data.empty:
            return self.vocs.random_inputs(self.options.n_initial)

        else:
            bounds = torch.tensor(self.vocs.bounds, **self._tkwargs)

            # update internal model with internal data
            self.train_model(self.data)

            # generate starting points for optimization
            inputs, _ = self.get_training_data(self.data)
            batch_initial_points = sample_truncated_normal_perturbations(
                inputs[-1].unsqueeze(0),
                n_discrete_points=self.options.optim.raw_samples,
                sigma=0.1,
                bounds=bounds,
            )

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
        Returns a SingleTaskGP (or ModelList set of independent SingleTaskGPs
        depending on the number of outputs, if data is None

        """

        inputs, outputs = self.get_training_data(data)
        input_transform = Normalize(
            self.vocs.n_variables, bounds=torch.tensor(self.vocs.bounds)
        )
        outcome_transform = Standardize(self.vocs.n_outputs)

        # create a batched single task GP model to represent independent outputs
        train_X = inputs
        train_Y = outputs

        model = SingleTaskGP(
            train_X,
            train_Y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        if update_internal:
            self._model = model
        return model

    def get_training_data(self, data: pd.DataFrame) -> (torch.Tensor, torch.Tensor):
        """
        get training data from dataframe
        - transform constraint data into standard form where values < 0 imply
        feasibility


        """
        inputs, outputs = self.vocs.get_training_data(data)
        return torch.tensor(inputs, **self._tkwargs), torch.tensor(
            outputs, **self._tkwargs
        )

    def get_acquisition(self, model):
        """
        Returns a function that can be used to evaluate the acquisition function
        """
        # re-create sampler/objective from options
        self.sampler = SobolQMCNormalSampler(self.options.acq.monte_carlo_samples)
        self.objective = self._get_objective()

        # add proximal biasing if requested
        if self.options.acq.proximal_lengthscales is not None:
            n_lengthscales = len(self.options.acq.proximal_lengthscales)
            if n_lengthscales != self.vocs.n_variables:
                raise ValueError(
                    f"Number of proximal lengthscales ({n_lengthscales}) must match "
                    f"number of variables {self.vocs.n_variables}"
                )

            acq = ProximalAcquisitionFunction(
                self._get_acquisition(model),
                torch.tensor(self.options.acq.proximal_lengthscales, **self._tkwargs),
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
