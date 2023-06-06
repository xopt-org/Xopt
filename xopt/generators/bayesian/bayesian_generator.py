import logging
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict

import pandas as pd
import torch
from botorch.acquisition import qUpperConfidenceBound
from botorch.models.gpytorch import GPyTorchModel
from botorch.optim import optimize_acqf
from botorch.sampling import get_sampler
from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning
from gpytorch import Module
from pydantic import Field, validator

from xopt.generator import Generator
from xopt.generators.bayesian.base_model import ModelConstructor
from xopt.generators.bayesian.custom_botorch.constrained_acqusition import (
    ConstrainedMCAcquisitionFunction,
)
from xopt.generators.bayesian.custom_botorch.proximal import ProximalAcquisitionFunction
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.bayesian.objectives import (
    create_constraint_callables,
    create_mc_objective,
)
from xopt.generators.bayesian.options import AcquisitionOptions, OptimizationOptions
from xopt.generators.bayesian.turbo import TurboState
from xopt.generators.bayesian.utils import rectilinear_domain_union, set_botorch_weights

logger = logging.getLogger()


class BayesianGenerator(Generator, ABC):
    name = "base_bayesian_generator"
    optimization_options: OptimizationOptions = Field(
        OptimizationOptions(),
        description="options used to customize optimization of the acquisition function",
    )
    model: GPyTorchModel = Field(
        None, description="mdel used by the generator to perform optimization"
    )
    turbo_state: TurboState = Field(
        default=None, description="turbo state for trust-region BO"
    )
    use_cuda: bool = Field(False, description="flag to enable cuda usage if available")
    model_constructor: ModelConstructor = Field(
        StandardModelConstructor(), description="constructor used to generate model"
    )
    acquisition_options: AcquisitionOptions = Field(
        AcquisitionOptions(),
        description="options used to customize acquisition function computation",
    )

    @validator("acquisition_options")
    def check_acq_options(cls, value: Dict, values):
        if isinstance(value, dict):
            pl = value["proximal_lengthscales"]
        elif isinstance(value, AcquisitionOptions):
            pl = value.proximal_lengthscales
        else:
            pl = None

        if pl is not None and "vocs" in values:
            n_variables = values["vocs"].n_variables
            if len(pl) != n_variables:
                raise ValueError(
                    "number of proximal lengthscales must equal number of variables"
                )

        return value

    @validator("model_constructor", pre=True)
    def validate_model_constructor(cls, value):
        constructor_dict = {"standard": StandardModelConstructor}
        if value is None:
            value = StandardModelConstructor()
        elif isinstance(value, ModelConstructor):
            value = value
        elif isinstance(value, str):
            if value in constructor_dict:
                value = constructor_dict[value]()
            else:
                raise ValueError(f"{value} not found")
        elif isinstance(value, dict):
            name = value.pop("name")
            if name in constructor_dict:
                value = constructor_dict[name](**value)
            else:
                raise ValueError(f"{value} not found")

        return value

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # set up turbo if requested
        if self.optimization_options.use_turbo and self.turbo_state is None:
            self.turbo_state = TurboState(self.vocs)

    def add_data(self, new_data: pd.DataFrame):
        self.data = pd.concat([self.data, new_data], axis=0)

    def generate(self, n_candidates: int) -> pd.DataFrame:
        if n_candidates > 1 and not self.supports_batch_generation:
            raise NotImplementedError(
                "This Bayesian algorithm does not currently support parallel candidate "
                "generation"
            )

        # if no data exists raise error
        if self.data.empty:
            raise RuntimeError(
                "no data contained in generator, call `add_data` "
                "method to add data, see also `Xopt.random_evaluate()`"
            )

        else:
            # update internal model with internal data
            model = self.train_model(self.data)

            # update TurBO state if used
            if self.turbo_state is not None:
                self.turbo_state.update_state(self.data)

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

        _model = self.model_constructor.build_model_from_vocs(self.vocs, data)

        if update_internal:
            self.model = _model
        return _model

    def get_input_data(self, data):
        return torch.tensor(
            self.vocs.variable_data(data, "").to_numpy(), **self._tkwargs
        )

    def get_acquisition(self, model):
        """
        Returns a function that can be used to evaluate the acquisition function
        """
        if model is None:
            raise ValueError("model cannot be None")

        # get base acquisition function
        acq = self._get_acquisition(model)

        # add proximal biasing if requested
        if self.acquisition_options.proximal_lengthscales is not None:
            acq = ProximalAcquisitionFunction(
                acq,
                torch.tensor(
                    self.acquisition_options.proximal_lengthscales, **self._tkwargs
                ),
                transformed_weighting=self.acquisition_options.use_transformed_proximal_weights,
                beta=10.0,
            )

        return acq

    def optimize_acqf(self, acq_funct, bounds, n_candidates):
        # get candidates in real domain
        candidates, out = optimize_acqf(
            acq_function=acq_funct,
            bounds=bounds,
            q=n_candidates,
            raw_samples=self.optimization_options.raw_samples,
            num_restarts=self.optimization_options.num_restarts,
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
            raw_samples=self.optimization_options.raw_samples * 5,
            num_restarts=self.optimization_options.num_restarts * 5,
        )

        return self._process_candidates(result)

    def _process_candidates(self, candidates):
        logger.debug("Best candidate from optimize", candidates)
        return self.vocs.convert_numpy_to_inputs(candidates.detach().cpu().numpy())

    def _get_sampler(self, model):
        input_data = self.get_input_data(self.data)
        sampler = get_sampler(
            model.posterior(input_data),
            sample_shape=torch.Size([self.acquisition_options.monte_carlo_samples]),
        )
        return sampler

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
    def _tkwargs(self):
        # set device and data type for generator
        device = "cpu"
        if self.use_cuda:
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
        bounds = deepcopy(self._get_bounds())

        # if specified modify bounds to limit maximum travel distances
        if self.optimization_options.max_travel_distances is not None:
            max_travel_bounds = self.get_max_travel_distances_region(bounds)
            bounds = rectilinear_domain_union(bounds, max_travel_bounds)

        # if using turbo, update turbo state and set bounds according to turbo state
        if self.turbo_state is not None:
            # set the best value
            turbo_bounds = self.turbo_state.get_trust_region(self.model)
            bounds = rectilinear_domain_union(bounds, turbo_bounds)

        return bounds

    def get_max_travel_distances_region(self, bounds):
        """get region based on max travel distances and last observation"""
        if len(self.optimization_options.max_travel_distances) != bounds.shape[-1]:
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

        # bound lengths based on vocs for normalization
        lengths = self.vocs.bounds[1, :] - self.vocs.bounds[0, :]

        # get maximum travel distances
        max_travel_distances = (
            torch.tensor(
                self.optimization_options.max_travel_distances, **self._tkwargs
            )
            * lengths
        )
        max_travel_bounds = torch.stack(
            (last_point - max_travel_distances, last_point + max_travel_distances)
        )

        return max_travel_bounds

    def _check_options(self, options):
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


class MultiObjectiveBayesianGenerator(BayesianGenerator, ABC):
    name = "multi_objective_bayesian_generator"
    reference_point: Dict[str, float] = Field(
        description="dict specifying reference point for multi-objective optimization"
    )

    supports_multi_objective = True

    @property
    def torch_reference_point(self):
        pt = []
        for name in self.vocs.objective_names:
            ref_val = self.reference_point[name]
            if self.vocs.objectives[name] == "MINIMIZE":
                pt += [-ref_val]
            elif self.vocs.objectives[name] == "MAXIMIZE":
                pt += [ref_val]
            else:
                raise ValueError(
                    f"objective type {self.vocs.objectives[name]} not\
                    supported"
                )

        return torch.tensor(pt, **self._tkwargs)

    def calculate_hypervolume(self):
        """compute hypervolume given data"""
        objective_data = torch.tensor(
            self.vocs.objective_data(self.data, return_raw=True).to_numpy()
        )

        # hypervolume must only take into account feasible data points
        if self.vocs.n_constraints > 0:
            objective_data = objective_data[
                self.vocs.feasibility_data(self.data)["feasible"].to_list()
            ]

        n_objectives = self.vocs.n_objectives
        weights = torch.zeros(n_objectives)
        weights = set_botorch_weights(weights, self.vocs)
        objective_data = objective_data * weights

        # compute hypervolume
        bd = DominatedPartitioning(
            ref_point=self.torch_reference_point, Y=objective_data
        )
        volume = bd.compute_hypervolume().item()

        return volume


def _preprocess_generator_arguments(kwargs):
    vocs = kwargs.get("vocs")
    model_constructor = kwargs.pop("model_constructor", None)

    if model_constructor is None:
        model_constructor = StandardModelConstructor(vocs=vocs)
    elif isinstance(model_constructor, ModelConstructor):
        model_constructor = model_constructor
    else:
        # replace model_constructor dict with object
        name = model_constructor.pop("name")
        if name == "standard":
            model_constructor = StandardModelConstructor(vocs=vocs, **model_constructor)

    kwargs["model_constructor"] = model_constructor

    return kwargs
