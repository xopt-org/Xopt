import logging
import os
import time
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional

import pandas as pd
import torch
from botorch.acquisition import FixedFeatureAcquisitionFunction, qUpperConfidenceBound
from botorch.models.model import Model
from botorch.sampling import get_sampler, SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning
from gpytorch import Module
from pydantic import Field, field_validator, SerializeAsAny
from pydantic_core.core_schema import FieldValidationInfo
from torch import Tensor

from xopt.generator import Generator
from xopt.generators.bayesian.base_model import ModelConstructor
from xopt.generators.bayesian.custom_botorch.constrained_acquisition import (
    ConstrainedMCAcquisitionFunction,
)
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.bayesian.objectives import (
    create_constraint_callables,
    create_mc_objective,
)
from xopt.generators.bayesian.turbo import (
    OptimizeTurboController,
    SafetyTurboController,
    TurboController,
)
from xopt.generators.bayesian.utils import rectilinear_domain_union, set_botorch_weights
from xopt.generators.bayesian.visualize import visualize_generator_model
from xopt.numerical_optimizer import GridOptimizer, LBFGSOptimizer, NumericalOptimizer
from xopt.pydantic import decode_torch_module

logger = logging.getLogger()

# It seems pydantic v2 does not auto-register models anymore
# So one option is to have explicit unions for model subclasses
# The other is to define a descriminated union with name as key, but that stops name field from
# getting exported, and we want to keep it for readability
# See https://github.com/pydantic/pydantic/discussions/5785

# Update: using parent model now seems to work, keeping this just in case
# T_ModelConstructor = Union[StandardModelConstructor, TimeDependentModelConstructor]
# T_NumericalOptimizer = Union[LBFGSOptimizer, GridOptimizer]


class BayesianGenerator(Generator, ABC):
    name = "base_bayesian_generator"
    model: Optional[Model] = Field(
        None, description="botorch model used by the generator to perform optimization"
    )
    n_monte_carlo_samples: int = Field(
        128, description="number of monte carlo samples to use"
    )
    turbo_controller: SerializeAsAny[Optional[TurboController]] = Field(
        default=None, description="turbo controller for trust-region BO"
    )
    use_cuda: bool = Field(False, description="flag to enable cuda usage if available")
    gp_constructor: SerializeAsAny[ModelConstructor] = Field(
        StandardModelConstructor(), description="constructor used to generate model"
    )
    numerical_optimizer: SerializeAsAny[NumericalOptimizer] = Field(
        LBFGSOptimizer(),
        description="optimizer used to optimize the acquisition " "function",
    )
    max_travel_distances: Optional[List[float]] = Field(
        None,
        description="limits for travel distance between points in normalized space",
    )
    fixed_features: Optional[Dict[str, float]] = Field(
        None, description="fixed features used in Bayesian optimization"
    )
    computation_time: Optional[pd.DataFrame] = Field(
        None,
        description="data frame tracking computation time in seconds",
    )
    n_candidates: int = 1

    @field_validator("model", mode="before")
    def validate_torch_modules(cls, v):
        if isinstance(v, str):
            if v.startswith("base64:"):
                v = decode_torch_module(v)
            elif os.path.exists(v):
                v = torch.load(v)
        return v

    @field_validator("gp_constructor", mode="before")
    def validate_gp_constructor(cls, value):
        print(f"Verifying model {value}")
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

    @field_validator("numerical_optimizer", mode="before")
    def validate_numerical_optimizer(cls, value):
        optimizer_dict = {"grid": GridOptimizer, "LBFGS": LBFGSOptimizer}
        if value is None:
            value = LBFGSOptimizer()
        elif isinstance(value, NumericalOptimizer):
            pass
        elif isinstance(value, str):
            if value in optimizer_dict:
                value = optimizer_dict[value]()
            else:
                raise ValueError(f"{value} not found")
        elif isinstance(value, dict):
            name = value.pop("name")
            if name in optimizer_dict:
                value = optimizer_dict[name](**value)
            else:
                raise ValueError(f"{value} not found")
        return value

    @field_validator("turbo_controller", mode="before")
    def validate_turbo_controller(cls, value, info: FieldValidationInfo):
        """note default behavior is no use of turbo"""
        optimizer_dict = {
            "optimize": OptimizeTurboController,
            "safety": SafetyTurboController,
        }
        if isinstance(value, TurboController):
            pass
        elif isinstance(value, str):
            # create turbo controller from string input
            if value in optimizer_dict:
                value = optimizer_dict[value](info.data["vocs"])
            else:
                raise ValueError(
                    f"{value} not found, available values are "
                    f"{optimizer_dict.keys()}"
                )
        elif isinstance(value, dict):
            # create turbo controller from dict input
            if "name" not in value:
                raise ValueError("turbo input dict needs to have a `name` attribute")
            name = value.pop("name")
            if name in optimizer_dict:
                value = optimizer_dict[name](vocs=info.data["vocs"], **value)
            else:
                raise ValueError(
                    f"{value} not found, available values are "
                    f"{optimizer_dict.keys()}"
                )
        return value

    @field_validator("computation_time", mode="before")
    def validate_computation_time(cls, value):
        if isinstance(value, dict):
            value = pd.DataFrame(value)

        return value

    def add_data(self, new_data: pd.DataFrame):
        self.data = pd.concat([self.data, new_data], axis=0)

    def generate(self, n_candidates: int) -> list[dict]:
        self.n_candidates = n_candidates
        if n_candidates > 1 and not self.supports_batch_generation:
            raise NotImplementedError(
                "This Bayesian algorithm does not currently support parallel candidate "
                "generation"
            )

        # if no data exists raise error
        if self.data is None:
            raise RuntimeError(
                "no data contained in generator, call `add_data` "
                "method to add data, see also `Xopt.random_evaluate()`"
            )

        else:
            # dict to track runtimes
            timing_results = {}

            # update internal model with internal data
            start_time = time.perf_counter()
            model = self.train_model(self.data)
            timing_results["training"] = time.perf_counter() - start_time

            # propose candidates given model
            start_time = time.perf_counter()
            candidates = self.propose_candidates(model, n_candidates=n_candidates)
            timing_results["acquisition_optimization"] = (
                time.perf_counter() - start_time
            )

            # post process candidates
            result = self._process_candidates(candidates)

            # append timing results to dataframe (if it exists)
            if self.computation_time is not None:
                self.computation_time = pd.concat(
                    (
                        self.computation_time,
                        pd.DataFrame(timing_results, index=[0]),
                    ),
                    ignore_index=True,
                )
            else:
                self.computation_time = pd.DataFrame(timing_results, index=[0])

            return result.to_dict("records")

    def train_model(self, data: pd.DataFrame = None, update_internal=True) -> Module:
        """
        Returns a ModelListGP containing independent models for the objectives and
        constraints

        """
        if data is None:
            data = self.data
        if data.empty:
            raise ValueError("no data available to build model")

        # get input bounds
        variable_bounds = deepcopy(self.vocs.variables)

        # add fixed feature bounds if requested
        if self.fixed_features is not None:
            # get bounds for each fixed_feature (vocs bounds take precedent)
            for key in self.fixed_features:
                if key not in variable_bounds:
                    f_data = data[key]
                    bounds = [f_data.min(), f_data.max()]
                    if bounds[1] - bounds[0] < 1e-8:
                        bounds[1] = bounds[0] + 1e-8
                    variable_bounds[key] = bounds

        _model = self.gp_constructor.build_model(
            self.model_input_names,
            self.vocs.output_names,
            data,
            {name: variable_bounds[name] for name in self.model_input_names},
            **self._tkwargs,
        )

        if update_internal:
            self.model = _model
        return _model

    def propose_candidates(self, model, n_candidates=1):
        """
        given a GP model, propose candidates by numerically optimizing the
        acquisition function

        """
        # update TurBO state if used with the last `n_candidates` points
        if self.turbo_controller is not None:
            self.turbo_controller.update_state(self.data, n_candidates)

        # calculate optimization bounds
        bounds = self._get_optimization_bounds()

        # get acquisition function
        acq_funct = self.get_acquisition(model)

        # get candidates
        candidates = self.numerical_optimizer.optimize(acq_funct, bounds, n_candidates)
        return candidates

    def get_input_data(self, data):
        return torch.tensor(data[self.model_input_names].to_numpy(), **self._tkwargs)

    def get_acquisition(self, model):
        """
        Returns a function that can be used to evaluate the acquisition function
        """
        if model is None:
            raise ValueError("model cannot be None")

        # get base acquisition function
        acq = self._get_acquisition(model)

        try:
            sampler = acq.sampler
        except AttributeError:
            sampler = SobolQMCNormalSampler(
                sample_shape=torch.Size([self.n_monte_carlo_samples])
            )

        # apply constraints if specified in vocs
        if len(self.vocs.constraints):
            acq = ConstrainedMCAcquisitionFunction(
                model, acq, self._get_constraint_callables(), sampler=sampler
            )

        # apply fixed features if specified in the generator
        if self.fixed_features is not None:
            # get input dim
            dim = len(self.model_input_names)
            columns = []
            values = []
            for name, value in self.fixed_features.items():
                columns += [self.model_input_names.index(name)]
                values += [value]

            acq = FixedFeatureAcquisitionFunction(
                acq_function=acq, d=dim, columns=columns, values=values
            )

        return acq

    def get_optimum(self):
        """select the best point(s) given by the
        model using the Posterior mean"""
        c_posterior_mean = ConstrainedMCAcquisitionFunction(
            self.model,
            qUpperConfidenceBound(
                model=self.model, beta=0.0, objective=self._get_objective()
            ),
            self._get_constraint_callables(),
        )

        result = self.numerical_optimizer.optimize(
            c_posterior_mean, self._get_bounds(), 1
        )

        return self._process_candidates(result)

    def visualize_model(self, **kwargs):
        """displays the GP models"""
        return visualize_generator_model(self, **kwargs)

    def _process_candidates(self, candidates: Tensor):
        """process pytorch candidates from optimizing the acquisition function"""
        logger.debug("Best candidate from optimize", candidates)

        if self.fixed_features is not None:
            results = pd.DataFrame(
                candidates.detach().cpu().numpy(), columns=self._candidate_names
            )
            for name, val in self.fixed_features.items():
                results[name] = val

        else:
            results = self.vocs.convert_numpy_to_inputs(
                candidates.detach().cpu().numpy(), include_constants=False
            )

        return results

    def _get_sampler(self, model):
        input_data = self.get_input_data(self.data)
        sampler = get_sampler(
            model.posterior(input_data),
            sample_shape=torch.Size([self.n_monte_carlo_samples]),
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

    @property
    def model_input_names(self):
        """variable names corresponding to trained model"""
        variable_names = self.vocs.variable_names
        if self.fixed_features is not None:
            for name, val in self.fixed_features.items():
                if name not in variable_names:
                    variable_names += [name]
        return variable_names

    @property
    def _candidate_names(self):
        """variable names corresponding to generated candidates"""
        variable_names = self.vocs.variable_names
        if self.fixed_features is not None:
            for name in self.fixed_features:
                if name in variable_names:
                    variable_names.remove(name)
        return variable_names

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
        if self.max_travel_distances is not None:
            max_travel_bounds = self._get_max_travel_distances_region(bounds)
            bounds = rectilinear_domain_union(bounds, max_travel_bounds)

        # if using turbo, update turbo state and set bounds according to turbo state
        if self.turbo_controller is not None:
            # set the best value
            turbo_bounds = self.turbo_controller.get_trust_region(self.model)
            bounds = rectilinear_domain_union(bounds, turbo_bounds)

        # if fixed features key is in vocs then we need to remove the bounds
        # associated with that key
        if self.fixed_features is not None:
            # grab variable name indices that are NOT in fixed features
            indicies = []
            for idx, name in enumerate(self.vocs.variable_names):
                if name not in self.fixed_features:
                    indicies += [idx]

            # grab indexed bounds
            bounds = bounds.T[indicies].T

        return bounds

    def _get_max_travel_distances_region(self, bounds):
        """get region based on max travel distances and last observation"""
        if len(self.max_travel_distances) != bounds.shape[-1]:
            raise ValueError(
                f"length of max_travel_distances must match the number of "
                f"variables {bounds.shape[-1]}"
            )

        # get last point
        if self.data is None:
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
            torch.tensor(self.max_travel_distances, **self._tkwargs) * lengths
        )
        max_travel_bounds = torch.stack(
            (last_point - max_travel_distances, last_point + max_travel_distances)
        )

        return max_travel_bounds


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
