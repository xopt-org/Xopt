import logging
from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd
import torch
from botorch.models import ModelListGP
from botorch.optim import optimize_acqf
from botorch.optim.initializers import sample_truncated_normal_perturbations
from botorch.sampling import SobolQMCNormalSampler
from gpytorch import Module

from xopt.generator import Generator
from xopt.generators.bayesian.options import BayesianOptions
from xopt.generators.bayesian.custom_botorch.proximal import ProximalAcquisitionFunction
from xopt.vocs import VOCS

logger = logging.getLogger()


class BayesianGenerator(Generator, ABC):
    def __init__(self, vocs: VOCS, options: BayesianOptions = None):
        options = options or BayesianOptions()
        if not isinstance(options, BayesianOptions):
            raise ValueError("options must be of type BayesianOptions")

        super().__init__(vocs, options)

        self._model = None
        self._acquisition = None
        self.sampler = SobolQMCNormalSampler(self.options.acq.monte_carlo_samples)
        self.objective = self._get_objective()

        self._tkwargs = {"dtype": torch.double, "device": "cpu"}

    @staticmethod
    def default_options() -> BayesianOptions:
        return BayesianOptions()

    def add_data(self, new_data: pd.DataFrame):
        self.data = pd.concat([self.data, new_data], axis=0)

    def generate(self, n_candidates: int) -> List[Dict]:
        if n_candidates > 1:
            raise NotImplementedError(
                "Bayesian algorithms don't currently support parallel candidate "
                "generation"
            )

        # if no data exists use random generator to generate candidates
        if self.data.empty:
            return self.vocs.random_inputs(self.options.n_initial)

        else:
            bounds = self._get_bounds()

            # update internal model with internal data
            self.train_model(self.data)

            if self.options.optim.use_nearby_initial_points:
                # generate starting points for optimization (note in real domain)
                inputs = self.get_input_data(self.data)
                batch_initial_points = sample_truncated_normal_perturbations(
                    inputs[-1].unsqueeze(0),
                    n_discrete_points=self.options.optim.raw_samples,
                    sigma=0.5,
                    bounds=bounds,
                ).unsqueeze(-2)
                raw_samples = None
            else:
                batch_initial_points = None
                raw_samples = self.options.optim.raw_samples

            acq_funct = self.get_acquisition(self._model)

            # get candidates in real domain
            candidates, out = optimize_acqf(
                acq_function=acq_funct,
                bounds=bounds,
                q=n_candidates,
                raw_samples=raw_samples,
                batch_initial_conditions=batch_initial_points,
                num_restarts=self.options.optim.num_restarts,
            )
            logger.debug("Best candidate from optimize", candidates, out)
            return self.vocs.convert_numpy_to_inputs(candidates.detach().numpy())

    def train_model(self, data: pd.DataFrame = None, update_internal=True) -> Module:
        """
        Returns a ModelListGP containing independent models for the objectives and
        constraints

        """
        if data is None:
            data = self.data

        # drop nans
        valid_data = data[
            pd.unique(self.vocs.variable_names + self.vocs.output_names)
        ].dropna()

        kwargs = self.options.model.kwargs.dict()

        _model = self.options.model.function(valid_data, self.vocs, **kwargs)

        # validate returned model
        self._validate_model(_model)

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

        # get base acquisition function
        acq = self._get_acquisition(model)

        # add proximal biasing if requested
        if self.options.acq.proximal_lengthscales is not None:
            acq = ProximalAcquisitionFunction(
                acq,
                torch.tensor(self.options.acq.proximal_lengthscales, **self._tkwargs),
                transformed_weighting=self.options.acq.use_transformed_proximal_weights,
                beta=10.0
            )

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

    def _get_bounds(self):
        bounds = torch.tensor(self.vocs.bounds, **self._tkwargs)
        # if specified modify bounds to limit maximum travel distances
        if self.options.optim.max_travel_distances is not None:
            if len(self.options.optim.max_travel_distances) != len(bounds):
                raise ValueError(
                    "max_travel_distances must be of length {}".format(len(bounds))
                )

            # get last point
            if self.data.empty:
                raise ValueError(
                    "No data exists to specify max_travel_distances "
                    "from, add data first to use during BO"
                )
            last_point = torch.tensor(
                self.data[self.vocs.variable_names].iloc[-1].to_numpy(), **self._tkwargs
            )

            # bound lengths for normalization
            lengths = bounds[1, :] - bounds[0, :]

            # get maximum travel distances
            max_travel_distances = (
                torch.tensor(self.options.optim.max_travel_distances, **self._tkwargs)
                * lengths
            )
            bounds[0, :] = torch.max(
                bounds[0, :],
                last_point - max_travel_distances,
            )
            bounds[1, :] = torch.min(
                bounds[1, :],
                last_point + max_travel_distances,
            )
        return bounds

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

    def _validate_model(self, model):
        if not isinstance(model, ModelListGP):
            raise ValueError("model must be ModelListGP object")
