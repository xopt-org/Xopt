import logging
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List

import pandas as pd
import torch
from botorch.acquisition import qUpperConfidenceBound
from botorch.models.model import Model
from botorch.sampling import get_sampler, SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning
from gpytorch import Module
from pydantic import Field, validator

from xopt.generator import Generator
from xopt.generators.bayesian.base_model import ModelConstructor
from xopt.generators.bayesian.custom_botorch.constrained_acqusition import (
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
from xopt.numerical_optimizer import GridOptimizer, LBFGSOptimizer, NumericalOptimizer

logger = logging.getLogger()


class BayesianGenerator(Generator, ABC):
    name = "base_bayesian_generator"
    model: Model = Field(
        None, description="botorch model used by the generator to perform optimization"
    )
    n_monte_carlo_samples = Field(
        128, description="number of monte carlo samples to use"
    )
    turbo_controller: TurboController = Field(
        default=None, description="turbo controller for trust-region BO"
    )
    use_cuda: bool = Field(False, description="flag to enable cuda usage if available")
    model_constructor: ModelConstructor = Field(
        StandardModelConstructor(), description="constructor used to generate model"
    )
    numerical_optimizer: NumericalOptimizer = Field(
        LBFGSOptimizer(),
        description="optimizer used to optimize the acquisition " "function",
    )
    max_travel_distances: List[float] = Field(
        None,
        description="limits for travel distance between points in normalized space",
    )

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

    @validator("numerical_optimizer", pre=True)
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

    @validator("turbo_controller", pre=True)
    def validate_turbo_controller(cls, value, values):
        """note default behavior is no use of turbo"""
        optimizer_dict = {
            "optimize": OptimizeTurboController(values["vocs"]),
            "safety": SafetyTurboController(values["vocs"]),
        }
        if isinstance(value, TurboController):
            pass
        elif isinstance(value, str):
            if value in optimizer_dict:
                value = optimizer_dict[value]
            else:
                raise ValueError(f"{value} not found")
        elif isinstance(value, dict):
            name = value.pop("name")
            if name in optimizer_dict:
                value = optimizer_dict[name](**value)
            else:
                raise ValueError(f"{value} not found")
        return value

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
            if self.turbo_controller is not None:
                self.turbo_controller.update_state(self.data)

            # calculate optimization bounds
            bounds = self._get_optimization_bounds()

            # get acquisition function
            acq_funct = self.get_acquisition(model)

            # get candidates
            candidates = self.numerical_optimizer.optimize(
                acq_funct, bounds, n_candidates
            )

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

        _model = self.model_constructor.build_model_from_vocs(
            self.vocs, data, **self._tkwargs
        )

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

        return acq

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

        result = self.numerical_optimizer.optimize(
            c_posterior_mean, self._get_bounds(), 1
        )

        return self._process_candidates(result)

    def _process_candidates(self, candidates):
        logger.debug("Best candidate from optimize", candidates)
        return self.vocs.convert_numpy_to_inputs(candidates.detach().cpu().numpy())

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
            max_travel_bounds = self.get_max_travel_distances_region(bounds)
            bounds = rectilinear_domain_union(bounds, max_travel_bounds)

        # if using turbo, update turbo state and set bounds according to turbo state
        if self.turbo_controller is not None:
            # set the best value
            turbo_bounds = self.turbo_controller.get_trust_region(self.model)
            bounds = rectilinear_domain_union(bounds, turbo_bounds)

        return bounds

    def get_max_travel_distances_region(self, bounds):
        """get region based on max travel distances and last observation"""
        if len(self.max_travel_distances) != bounds.shape[-1]:
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
