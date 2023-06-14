import logging
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd
import torch
from botorch.acquisition import qUpperConfidenceBound
from botorch.models import ModelListGP
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.sampling.get_sampler import get_sampler
from gpytorch import Module

from xopt.generator import Generator
from xopt.generators.bayesian.custom_botorch.constrained_acqusition import (
    ConstrainedMCAcquisitionFunction,
)
from xopt.generators.bayesian.custom_botorch.proximal import ProximalAcquisitionFunction
from xopt.generators.bayesian.models.utils import get_model_constructor
from xopt.generators.bayesian.objectives import (
    create_constraint_callables,
    create_mc_objective,
)
from xopt.generators.bayesian.options import BayesianOptions
from xopt.generators.bayesian.turbo import get_trust_region, TurboState, update_state
from xopt.generators.bayesian.utils import rectilinear_domain_union
from xopt.vocs import VOCS

logger = logging.getLogger()


class BayesianGenerator(Generator, ABC):
    def __init__(
        self,
        vocs: VOCS,
        options: BayesianOptions = None,
        supports_batch_generation=False,
    ):
        options = options or BayesianOptions()
        if not isinstance(options, BayesianOptions):
            raise ValueError("options must be of type BayesianOptions")

        super().__init__(vocs, options)

        self._model = None
        self._acquisition = None
        self._trust_region = None
        self.supports_batch_generation = supports_batch_generation
        self.sampler = SobolQMCNormalSampler(self.options.acq.monte_carlo_samples)
        self.model_constructor = get_model_constructor(options.model)(
            self.vocs, options.model
        )

        # set up turbo if requested
        if self.options.optim.use_turbo:
            self.turbo_state = TurboState(self.vocs.n_variables, 1)
            self.first_call = True

    @staticmethod
    def default_options() -> BayesianOptions:
        return BayesianOptions()

    def add_data(self, new_data: pd.DataFrame):
        self.data = pd.concat([self.data, new_data], axis=0)

    def generate(self, n_candidates: int) -> List[Dict]:
        if n_candidates > 1 and not self.supports_batch_generation:
            raise NotImplementedError(
                "This Bayesian algorithm does not currently support parallel candidate "
                "generation"
            )

        # if no data exists use random generator to generate candidates
        if self.data.empty:
            return self.vocs.random_inputs(self.options.n_initial)

        else:
            # update internal model with internal data
            model = self.train_model(self.data)

            # calculate optimization bounds
            bounds = self._get_optimization_bounds()

            # get acquisition function
            acq_funct = self.get_acquisition(model)

            # get candidates
            candidates = self.optimize_acqf(acq_funct, bounds, n_candidates)

            # post process candidates
            result = self._process_candidates(candidates)
            return result

    def train_model(self, data: pd.DataFrame = None, update_internal=True) -> Module:
        """
        Returns a ModelListGP containing independent models for the objectives and
        constraints

        """
        if data is None:
            data = self.data
        if data.empty:
            raise ValueError("no data available to build model")

        _model = self.model_constructor.build_model(data, self._tkwargs)

        # validate returned model
        self._validate_model(_model)

        if update_internal:
            self._model = _model
        return _model

    def get_input_data(self, data):
        return torch.tensor(
            self.vocs.variable_data(data, "").to_numpy(), **self._tkwargs
        )

    def get_acquisition(self, model):
        """
        Returns a function that can be used to evaluate the acquisition function
        """

        # need a sampler for botorch > 0.8
        # get input data
        input_data = self.get_input_data(self.data)
        self.sampler = get_sampler(
            model.posterior(input_data),
            sample_shape=torch.Size([self.options.acq.monte_carlo_samples]),
        )

        # get base acquisition function
        acq = self._get_acquisition(model)

        # add proximal biasing if requested
        if self.options.acq.proximal_lengthscales is not None:
            acq = ProximalAcquisitionFunction(
                acq,
                torch.tensor(self.options.acq.proximal_lengthscales, **self._tkwargs),
                transformed_weighting=self.options.acq.use_transformed_proximal_weights,
                beta=10.0,
            )

        return acq

    def optimize_acqf(self, acq_funct, bounds, n_candidates):
        # get candidates in real domain
        candidates, out = optimize_acqf(
            acq_function=acq_funct,
            bounds=bounds,
            q=n_candidates,
            raw_samples=self.options.optim.raw_samples,
            num_restarts=self.options.optim.num_restarts,
        )
        return candidates

    def get_optimum(self):
        """select the best point(s) (for multi-objective generators, given by the
        model using the Posterior mean"""
        c_posterior_mean = ConstrainedMCAcquisitionFunction(
            self.model,
            qUpperConfidenceBound(
                model=self.model, beta=0.0, objective=self._get_objective()
            ),
            self._get_constraint_callables(),
        )

        result, out = optimize_acqf(
            acq_function=c_posterior_mean,
            bounds=self._get_bounds(),
            q=1,
            raw_samples=self.options.optim.raw_samples * 5,
            num_restarts=self.options.optim.num_restarts * 5,
        )

        return self._process_candidates(result)

    def _process_candidates(self, candidates):
        logger.debug("Best candidate from optimize", candidates)
        return self.vocs.convert_numpy_to_inputs(candidates.detach().cpu().numpy())

    @abstractmethod
    def _get_acquisition(self, model):
        pass

    def _get_objective(self):
        """return default objective (scalar objective) determined by vocs"""
        return create_mc_objective(self.vocs, self._tkwargs)

    def _get_constraint_callables(self):
        """return default objective (scalar objective) determined by vocs"""
        constraint_callables = create_constraint_callables(self.vocs)
        if len(constraint_callables) == 0:
            constraint_callables = None
        return constraint_callables

    @property
    def model(self):
        if self._model is None:
            self.train_model(self.data)
        return self._model

    @property
    def _tkwargs(self):
        # set device and data type for generator
        device = "cpu"
        if self.options.use_cuda:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                warnings.warn(
                    "Cuda requested in generator options but not found on "
                    "machine! Using CPU instead"
                )

        return {"dtype": torch.double, "device": device}

    def _get_bounds(self):
        """convert bounds from vocs to torch tensors"""
        return torch.tensor(self.vocs.bounds, **self._tkwargs)

    def _get_optimization_bounds(self):
        """
        gets optimization bounds based on the union of several domains
        - if use_turbo is True include trust region
        - if max_travel_distances is not None limit max travel distance

        """
        bounds = self._get_bounds()

        # if using turbo, update turbo state and set bounds according to turbo state
        if self.options.optim.use_turbo:
            bounds = self.get_trust_region(bounds)

        # if specified modify bounds to limit maximum travel distances
        if self.options.optim.max_travel_distances is not None:
            bounds = self.get_max_travel_distances_region(bounds)

        return bounds

    def get_trust_region(self, bounds):
        """get trust region based on turbo_state and last observation"""
        if self.options.optim.use_turbo:
            objective_data = self.vocs.objective_data(self.data, "")

            # if this is the first time we are updating the state use the best f
            # instead of the last f
            if self.first_call:
                y_last = torch.tensor(objective_data.min().to_numpy(), **self._tkwargs)
                self.first_call = False
            else:
                y_last = torch.tensor(
                    objective_data.iloc[-1].to_numpy(), **self._tkwargs
                )
            self.turbo_state = update_state(self.turbo_state, y_last)

            # calculate trust region and apply to base bounds
            trust_region = get_trust_region(
                self.vocs,
                self.model,
                bounds,
                self.data,
                self.turbo_state,
                self._tkwargs,
            )

            return rectilinear_domain_union(bounds, trust_region)
        else:
            raise RuntimeError(
                "cannot get trust region when `use_turbo` option is False"
            )

    def get_max_travel_distances_region(self, bounds):
        """get region based on max travel distances and last observation"""
        if len(self.options.optim.max_travel_distances) != bounds.shape[-1]:
            raise ValueError(
                f"length of max_travel_distances must match the number of "
                f"variables {bounds.shape[-1]}"
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
        max_travel_bounds = torch.stack(
            (last_point - max_travel_distances, last_point + max_travel_distances)
        )

        return rectilinear_domain_union(bounds, max_travel_bounds)

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
