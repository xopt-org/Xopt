import logging
import os
import time
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import (
    FixedFeatureAcquisitionFunction,
    qUpperConfidenceBound,
    AcquisitionFunction,
)
from botorch.models.model import Model
from botorch.sampling import get_sampler
from gpytorch import Module
from pydantic import (
    Field,
    field_validator,
    PositiveInt,
    SerializeAsAny,
)
from pydantic.fields import PrivateAttr, ModelPrivateAttr
from pydantic_core.core_schema import ValidationInfo
from torch import Tensor

from xopt.errors import XoptError, FeasibilityError
from xopt.generator import Generator
from xopt.generators.bayesian.base_model import ModelConstructor
from xopt.generators.bayesian.custom_botorch.constrained_acquisition import (
    ConstrainedMCAcquisitionFunction,
)
from xopt.generators.bayesian.custom_botorch.log_acquisition_function import (
    LogAcquisitionFunction,
)
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.bayesian.objectives import (
    create_constraint_callables,
    create_mc_objective,
    CustomXoptObjective,
)
from xopt.generators.bayesian.turbo import (
    TurboController,
)
from xopt.generators.bayesian.utils import (
    interpolate_points,
    rectilinear_domain_union,
    set_botorch_weights,
    validate_turbo_controller_base,
    compute_hypervolume_and_pf,
)
from xopt.generators.bayesian.visualize import visualize_generator_model
from xopt.numerical_optimizer import GridOptimizer, LBFGSOptimizer, NumericalOptimizer
from xopt.pydantic import decode_torch_module
from xopt.vocs import VOCS

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
    """Bayesian Generator for Bayesian Optimization.

    Attributes
    ----------
    name : str
        The name of the Bayesian Generator.

    model : Optional[Model]
        The BoTorch model used by the generator to perform optimization.

    n_monte_carlo_samples : int
        The number of Monte Carlo samples to use in the optimization process.

    turbo_controller : SerializeAsAny[Optional[TurboController]]
        The Turbo Controller for trust-region Bayesian Optimization.

    use_cuda : bool
        A flag to enable or disable CUDA usage if available.

    gp_constructor : SerializeAsAny[ModelConstructor]
        The constructor used to generate the model for Bayesian Optimization.

    numerical_optimizer : SerializeAsAny[NumericalOptimizer]
        The optimizer used to optimize the acquisition function in Bayesian Optimization.

    max_travel_distances : Optional[List[float]]
        The limits for travel distances between points in normalized space.

    fixed_features : Optional[Dict[str, float]]
        The fixed features used in Bayesian Optimization.

    computation_time : Optional[pd.DataFrame]
        A data frame tracking computation time in seconds.

    n_interpolate_samples: Optional[PositiveInt]
        Number of interpolation points to generate between last observation and next
        observation, requires n_candidates to be 1.

    n_candidates : int
        The number of candidates to generate in each optimization step.

    Methods
    -------
    generate(self, n_candidates: int) -> List[Dict]:
        Generate candidates for Bayesian Optimization.

    add_data(self, new_data: pd.DataFrame):
        Add new data to the generator for Bayesian Optimization.

    train_model(self, data: pd.DataFrame = None, update_internal=True) -> Module:
        Train a Bayesian model for Bayesian Optimization.

    propose_candidates(self, model: Module, n_candidates: int = 1) -> Tensor:
        Propose candidates for Bayesian Optimization.

    get_input_data(self, data: pd.DataFrame) -> torch.Tensor:
        Get input data in torch.Tensor format.

    get_acquisition(self, model: Module) -> AcquisitionFunction:
        Get the acquisition function for Bayesian Optimization.

    """

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
        description="optimizer used to optimize the acquisition function",
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
    custom_objective: Optional[CustomXoptObjective] = Field(
        None,
        description="custom objective for optimization, replaces objective specified by VOCS",
    )
    n_interpolate_points: Optional[PositiveInt] = None

    n_candidates: int = 1

    _compatible_turbo_controllers: Optional[list[type[TurboController]]] = PrivateAttr(
        default=None
    )
    _compatible_numerical_optimizers: list[type[NumericalOptimizer]] = PrivateAttr(
        default=[LBFGSOptimizer, GridOptimizer]
    )

    @classmethod
    def get_compatible_turbo_controllers(cls) -> list[type[TurboController] | None]:
        compatible = cls._compatible_turbo_controllers
        if compatible is None:
            return [None]

        compatible_list: list[type[TurboController] | None] = []
        # If it's a ModelPrivateAttr, get the default value
        if isinstance(compatible, ModelPrivateAttr):
            compatible = compatible.get_default()
        # Defensive: ensure it's a list
        compatible_list = list(compatible) if compatible is not None else []
        if None not in compatible_list:
            compatible_list.append(None)
        return compatible_list

    @classmethod
    def get_compatible_numerical_optimizers(
        cls,
    ) -> list[type[NumericalOptimizer]]:
        compatible = cast(ModelPrivateAttr, cls._compatible_numerical_optimizers)
        return compatible.get_default()

    @field_validator("model", mode="before")
    @classmethod
    def validate_torch_modules(cls, value: Any):
        if isinstance(value, str):
            if value.startswith("base64:"):
                value = decode_torch_module(value)
            elif os.path.exists(value):
                value = torch.load(value, weights_only=False)
        return value

    @field_validator("gp_constructor", mode="before")
    @classmethod
    def validate_gp_constructor(cls, value: Any):
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
            _value = cast(dict[str, Any], value)
            name = _value.pop("name", "")
            if name in constructor_dict:
                value = constructor_dict[name](**_value)
            else:
                raise ValueError(f"{value} not found")

        return value

    @field_validator("numerical_optimizer", mode="before")
    @classmethod
    def validate_numerical_optimizer(cls, value: Any):
        optimizer_dict: dict[str, type[NumericalOptimizer]] = {
            "grid": GridOptimizer,
            "LBFGS": LBFGSOptimizer,
        }
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
            _value = cast(dict[str, Any], value)
            name: str = _value.pop("name", "")
            if name in optimizer_dict:
                value = optimizer_dict[name](**_value)
            else:
                raise ValueError(f"{_value} not found")
        else:
            raise ValueError(f"{value} not recognized as NumericalOptimizer")
        return value

    @field_validator("turbo_controller", mode="before")
    @classmethod
    def validate_turbo_controller(cls, value: Any, info: ValidationInfo):
        """note default behavior is no use of turbo"""
        if value is None:
            return value

        if cls._compatible_turbo_controllers is None:
            raise ValueError("no turbo controllers are compatible with this generator")

        compatible_turbo_controllers = [
            turbo_controller
            for turbo_controller in cls.get_compatible_turbo_controllers()
            if turbo_controller is not None
        ]

        if len(compatible_turbo_controllers) == 0:
            raise ValueError("no turbo controllers are compatible with this generator")
        else:
            return validate_turbo_controller_base(
                value, compatible_turbo_controllers, info
            )

    @field_validator("computation_time", mode="before")
    @classmethod
    def validate_computation_time(cls, value: Any):
        if value is None:
            return value
        elif isinstance(value, pd.DataFrame):
            return value
        elif isinstance(value, dict):
            value = pd.DataFrame(value)
        else:
            raise ValueError(
                "computation_time must be a pandas DataFrame, dict, or None"
            )

        return value

    def add_data(self, new_data: pd.DataFrame):
        """
        Add new data to the generator for Bayesian Optimization.

        Parameters
        ----------
        new_data : pd.DataFrame
            The new data to be added to the generator.

        Notes
        -----
        This method appends the new data to the existing data in the generator.
        """
        self.data = pd.concat([self.data, new_data], axis=0)

    def generate(self, n_candidates: int):
        """
        Generate candidates using Bayesian Optimization.

        Parameters
        ----------
        n_candidates : int
            The number of candidates to generate in each optimization step.

        Returns
        -------
        List[Dict]
            A list of dictionaries containing the generated candidates.

        Raises
        ------
        NotImplementedError
            If the number of candidates is greater than 1, and the generator does not
            support batch candidate generation.

        RuntimeError
            If no data is contained in the generator, the 'add_data' method should be
            called to add data before generating candidates.

        Notes
        -----
        This method generates candidates for Bayesian Optimization based on the
        provided number of candidates. It updates the internal model with the current
        data and calculates the candidates by optimizing the acquisition function.
        The method returns the generated candidates in the form of a list of dictionaries.
        """

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
            model = self.train_model(self.get_training_data(self.data))
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

            if self.n_interpolate_points is not None:
                if self.n_candidates > 1:
                    raise RuntimeError(
                        "cannot generate interpolated points for "
                        "multiple candidate generation"
                    )
                else:
                    assert len(result) == 1
                    result = interpolate_points(
                        pd.concat(
                            (self.data.iloc[-1:][self.vocs.variable_names], result),
                            axis=0,
                            ignore_index=True,
                        ),
                        num_points=self.n_interpolate_points,
                    )

            return result.to_dict("records")

    def train_model(
        self, data: pd.DataFrame | None = None, update_internal: bool = True
    ) -> Module:
        """
        Train a Bayesian model for Bayesian Optimization.

        Parameters
        ----------
        data : pd.DataFrame, optional
            The data to be used for training the model. If not provided, the internal
            data of the generator is used.
        update_internal : bool, optional
            Flag to indicate whether to update the internal model of the generator
            with the trained model (default is True).

        Returns
        -------
        Module
            The trained Bayesian model.

        Raises
        ------
        ValueError
            If no data is available to build the model.

        Notes
        -----
        This method trains a Bayesian model using the provided data or the internal
        data of the generator. It updates the internal model with the trained model
        if the 'update_internal' flag is set to True.
        """
        if data is None:
            data = self.get_training_data(self.data)
        if data.empty:
            raise ValueError("no data available to build model")

        # get input bounds
        variable_bounds = deepcopy(self.vocs.variables)

        # if turbo restrict points is true then set the bounds to the trust region
        # bounds
        if self.turbo_controller is not None:
            if self.turbo_controller.restrict_model_data:
                variable_bounds = dict(
                    zip(
                        self.vocs.variable_names,
                        self.turbo_controller.get_trust_region(self).numpy().T,
                    )
                )

        # add fixed feature bounds if requested
        if self.fixed_features is not None:
            # get bounds for each fixed_feature (vocs bounds take precedent)
            for key in self.fixed_features:
                if key not in variable_bounds:
                    if key not in data:
                        raise KeyError(
                            "generator data needs to contain fixed feature "
                            f"column name `{key}`"
                        )
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
            **self.tkwargs,
        )

        if update_internal:
            self.model = _model

        return _model

    def propose_candidates(self, model: Module, n_candidates: int = 1) -> Tensor:
        """
        Propose candidates using Bayesian Optimization.

        Parameters
        ----------
        model : Module
            The trained Bayesian model.
        n_candidates : int, optional
            The number of candidates to propose (default is 1).

        Returns
        -------
        Tensor
            A tensor containing the proposed candidates.

        Notes
        -----
        This method proposes candidates for Bayesian Optimization by numerically
        optimizing the acquisition function using the trained model. It updates the
        state of the Turbo controller if used and calculates the optimization bounds.
        """
        # update TurBO state if used with the last `n_candidates` points
        if self.turbo_controller is not None:
            self.turbo_controller.update_state(self, n_candidates)

        # calculate optimization bounds
        bounds = self._get_optimization_bounds()

        # get acquisition function
        acq_funct = self.get_acquisition(model)

        # get initial candidates to start acquisition function optimization
        initial_points = self._get_initial_conditions(n_candidates)

        # get candidates -- grid optimizer does not support batch_initial_conditions
        if isinstance(self.numerical_optimizer, GridOptimizer):
            candidates = self.numerical_optimizer.optimize(
                acq_funct, bounds, n_candidates
            )
        else:
            candidates = self.numerical_optimizer.optimize(
                acq_funct, bounds, n_candidates, batch_initial_conditions=initial_points
            )
        return candidates

    def get_training_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get training data used to train the GP model.

        If a turbo controller is specified with the flag `restrict_model_data` this
        will return a subset of data that is inside the trust region.

        Parameters
        ----------
        data : pd.DataFrame
            The data in the form of a pandas DataFrame.

        Returns
        -------
        data : pd.DataFrame
            A subset of data used to train the model form of a pandas DataFrame.

        """
        if self.turbo_controller is not None:
            if self.turbo_controller.restrict_model_data:
                data = self.turbo_controller.get_data_in_trust_region(data, self)
                if data.empty:
                    raise FeasibilityError(
                        "No training data available to build model, because ",
                        "no points in the dataset are within the TuRBO trust region. ",
                    )
        return data

    def get_input_data(self, data: pd.DataFrame) -> torch.Tensor:
        """
        Convert input data to a torch tensor.

        Parameters
        ----------
        data : pd.DataFrame
            The input data in the form of a pandas DataFrame.

        Returns
        -------
        torch.Tensor
            A torch tensor containing the input data.

        Notes
        -----
        This method takes a pandas DataFrame as input data and converts it into a
        torch tensor. It specifically selects columns corresponding to the model's
        input names (variables), and the resulting tensor is configured with the data
        type and device settings from the generator.
        """
        return torch.tensor(data[self.model_input_names].to_numpy(), **self.tkwargs)

    def get_acquisition(self, model: Module) -> AcquisitionFunction:
        """
        Define the acquisition function based on the given GP model.

        Lives on target device specified by tkwargs / use_cuda.

        Parameters
        ----------
        model : Module
            The BoTorch model to be used for generating the acquisition function.

        Returns
        -------
        acqusition_function : AcquisitionFunction

        Raises
        ------
        ValueError
            If the provided 'model' is None. A valid model is required to create the
            acquisition function.
        """
        if model is None:
            raise ValueError("model cannot be None")

        # get base acquisition function
        acq = self._get_acquisition(model)

        # apply constraints if specified in vocs
        # TODO: replace with direct constrainted acquisition function calls
        # see SampleReducingMCAcquisitionFunction in botorch for rationale
        if len(self.vocs.constraints):
            try:
                sampler = acq.sampler
            except AttributeError:
                sampler = self._get_sampler(model)

            acq = ConstrainedMCAcquisitionFunction(
                model, acq, self._get_constraint_callables(), sampler=sampler
            )

            # log transform the result to handle the constraints
            acq = LogAcquisitionFunction(acq)

        acq = self._apply_fixed_features(acq)
        acq = acq.to(**self.tkwargs)
        return acq

    def get_optimum(self):
        """select the best point(s) given by the
        model using the Posterior mean"""
        acq = qUpperConfidenceBound(
            model=self.model, beta=0.0, objective=self._get_objective()
        )
        if len(self.vocs.constraints):
            acq = ConstrainedMCAcquisitionFunction(
                self.model,
                acq,
                self._get_constraint_callables(),
                sampler=self._get_sampler(self.model),
            )
        bounds = self._get_bounds()

        if self.fixed_features is not None:
            acq = self._apply_fixed_features(acq)

            indices = []
            for idx, name in enumerate(self.vocs.variable_names):
                if name not in self.fixed_features:
                    indices += [idx]

            bounds = bounds[:, indices]

        bounds = bounds.to(**self.tkwargs)
        acq = acq.to(**self.tkwargs)

        # use default initial conditions for a global search
        result = self.numerical_optimizer.optimize(acq, bounds, 1)

        return self._process_candidates(result)

    def visualize_model(self, **kwargs):
        """Display GP model predictions for the selected output(s).

        The GP models are displayed with respect to the named variables. If None are given, the list of variables in
        vocs is used. Feasible samples are indicated with a filled orange "o", infeasible samples with a hollow
        red "o". Feasibility is calculated with respect to all constraints unless the selected output is a
        constraint itself, in which case only that one is considered.

        Parameters
        ----------
        **kwargs: dict, optional
            Supported keyword arguments:
            - output_names : List[str]
                Outputs for which the GP models are displayed. Defaults to all outputs in vocs.
            - variable_names : List[str]
                The variables with respect to which the GP models are displayed (maximum of 2).
                Defaults to vocs.variable_names.
            - idx : int
                Index of the last sample to use. This also selects the point of reference in
                higher dimensions unless an explicit reference_point is given.
            - reference_point : dict
                Reference point determining the value of variables in vocs.variable_names, but not in variable_names
                (slice plots in higher dimensions). Defaults to last used sample.
            - show_samples : bool, optional
                Whether samples are shown.
            - show_prior_mean : bool, optional
                Whether the prior mean is shown.
            - show_feasibility : bool, optional
                Whether the feasibility region is shown.
            - show_acquisition : bool, optional
                Whether the acquisition function is computed and shown (only if acquisition function is not None).
            - n_grid : int, optional
                Number of grid points per dimension used to display the model predictions.
            - axes : Axes, optional
                Axes object used for plotting.
            - exponentiate : bool, optional
                Flag to exponentiate acquisition function before plotting.

        Returns
        -------
        result : tuple
            The matplotlib figure and axes objects.
        """
        return visualize_generator_model(self, **kwargs)

    def _get_initial_conditions(self, n_candidates=1) -> Union[Tensor, None]:
        """overwrite if algorithm should specifiy initial candidates for optimizing
        the acquisition function"""
        return None

    def _process_candidates(self, candidates: Tensor):
        """process pytorch candidates from optimizing the acquisition function"""
        logger.debug(f"Best candidate from optimize {candidates}")

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
        """
        Return default objective (scalar objective) determined by vocs or if
        defined in custom_objective. Module is already on target device.
        """
        # check to make sure that if we specify a custom objective that no objectives
        # are specified in vocs
        if self.custom_objective is not None:
            if self.vocs.n_objectives:
                raise RuntimeError(
                    "cannot specify objectives in VOCS "
                    "and a custom objective for the generator at the "
                    "same time"
                )

            objective = self.custom_objective
        else:
            objective = create_mc_objective(self.vocs)

        return objective.to(**self.tkwargs)

    def _get_constraint_callables(self):
        """return constraint callable determined by vocs"""
        constraint_callables = create_constraint_callables(self.vocs)
        if len(constraint_callables) == 0:
            constraint_callables = None
        return constraint_callables

    def _apply_fixed_features(self, acq):
        """apply fixed features to the acquisition function if needed"""
        if self.fixed_features is not None:
            # get input dim
            dim = len(self.model_input_names)
            columns = []
            values = []
            for name, value in self.fixed_features.items():
                columns.append(self.model_input_names.index(name))
                values.append(value)

            # necessary because fixed feature acq must get tensor - it searches for dtype/device
            values = torch.tensor(values).to(**self.tkwargs)
            acq = FixedFeatureAcquisitionFunction(
                acq_function=acq, d=dim, columns=columns, values=values
            )
        return acq

    @property
    def tkwargs(self):
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

    def _get_bounds(self) -> torch.Tensor:
        """
        Convert bounds from vocs to torch tensors of shape 2 x d.

        Tensor stays on CPU
        """
        return torch.tensor(self.vocs.bounds)

    def _get_optimization_bounds(self):
        """
        Get optimization bounds based on the union of several domains.

        Returns
        -------
        torch.Tensor
            Tensor containing the optimized bounds.

        Notes
        -----
        This method calculates the optimization bounds based on several factors:

        - If 'max_travel_distances' is specified, the bounds are modified to limit
            the maximum travel distances between points in normalized space.
        - If 'turbo_controller' is not None, the bounds are updated according to the
            trust region specified by the controller.
        - If 'fixed_features' are included in the variable names from the VOCS,
            the bounds associated with those features are removed.

        """
        bounds = self._get_bounds()

        # if specified modify bounds to limit maximum travel distances
        if self.max_travel_distances is not None:
            max_travel_bounds = self._get_max_travel_distances_region(bounds)
            bounds = rectilinear_domain_union(bounds, max_travel_bounds)

        # if using turbo, update turbo state and set bounds according to turbo state
        if self.turbo_controller is not None:
            # set the best value
            turbo_bounds = self.turbo_controller.get_trust_region(self)
            bounds = rectilinear_domain_union(bounds, turbo_bounds)

        # if fixed features key is in vocs then we need to remove the bounds
        # associated with that key
        if self.fixed_features is not None:
            # grab variable name indices that are NOT in fixed features
            indices = []
            for idx, name in enumerate(self.vocs.variable_names):
                if name not in self.fixed_features:
                    indices += [idx]

            # grab indexed bounds
            bounds = bounds[:, indices]

        bounds = bounds.to(**self.tkwargs)
        return bounds

    def _get_max_travel_distances_region(self, bounds):
        """
        Calculate the region for maximum travel distances based on the current bounds
        and the last observation.

        Tensor stays on CPU.

        Parameters
        ----------
        bounds : torch.Tensor
            The optimization bounds based on the union of several domains.

        Returns
        -------
        torch.Tensor
            The bounds for the maximum travel distances region.

        Raises
        ------
        ValueError
            If the length of max_travel_distances does not match the number of
            variables in bounds.

        Notes
        -----
        This method calculates the region in which the next candidates for
        optimization should be generated based on the maximum travel distances
        specified. The region is centered around the last observation in the
        optimization space. The `max_travel_distances` parameter should be a list of
        maximum travel distances for each variable.

        """
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
        last_point = self.data[self.vocs.variable_names].iloc[-1].to_numpy()

        # bound lengths based on vocs for normalization
        vocs_bounds = self.vocs.bounds
        lengths = vocs_bounds[1, :] - vocs_bounds[0, :]

        # get maximum travel distances
        max_travel_distances = np.array(self.max_travel_distances) * lengths

        max_travel_bounds = np.stack(
            (last_point - max_travel_distances, last_point + max_travel_distances)
        )

        return torch.tensor(max_travel_bounds)


class MultiObjectiveBayesianGenerator(BayesianGenerator, ABC):
    name = "multi_objective_bayesian_generator"
    reference_point: dict[str, float] = Field(
        {},
        description="dict specifying reference point for multi-objective optimization",
        # validate_default=True,
    )
    pareto_front_history: Optional[pd.DataFrame] = Field(
        None,
        description="history of pareto front statistics every time points are added to the generator",
        exclude=True,
    )

    supports_multi_objective: bool = True

    @field_validator("pareto_front_history", mode="before")
    @classmethod
    def validate_pareto_front_history(cls, value: Any):
        return pd.DataFrame(value) if value is not None else None

    @field_validator("reference_point", mode="after")
    @classmethod
    def validate_reference_point(cls, value: dict[str, float], info: ValidationInfo):
        # set default reference point if not specified
        _vocs: VOCS | None = info.data.get("vocs", None)
        objective_names = _vocs.objective_names if _vocs is not None else []

        if set(value.keys()) != set(objective_names):
            raise XoptError("reference point must contain all objective names in vocs")

        return value

    @property
    def torch_reference_point(self):
        pt = []
        for name in self.vocs.objective_names:
            try:
                ref_val = self.reference_point[name]
            except KeyError:
                raise XoptError(
                    "need to specify reference point for the following "
                    f"objective {name}"
                )
            if self.vocs.objectives[name] == "MINIMIZE":
                pt += [-ref_val]
            elif self.vocs.objectives[name] == "MAXIMIZE":
                pt += [ref_val]
            else:
                raise ValueError(
                    f"objective type {self.vocs.objectives[name]} not\
                    supported"
                )

        return torch.tensor(pt, **self.tkwargs)

    def get_pareto_front_and_hypervolume(
        self,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, float]:
        """
        Get the pareto front and hypervolume of the current data.

        Returns
        -------
        pareto_front_variables : torch.Tensor
            The pareto front variable data.
        pareto_front_objectives : torch.Tensor
            The pareto front objective data.
        pareto_mask : torch.Tensor
            A mask indicating which points are part of the pareto front.
        hv : float
            The hypervolume of the pareto front.
        """

        # get scaled data
        # note that the objective data is scaled by +/- 1
        # based on maximization / minimization
        variable_data, objective_data, weights = self._get_scaled_data(data=self.data)

        # if there are no valid points skip PF calculation and return None
        if len(variable_data) == 0:
            return None, None, None, 0.0

        pareto_front_variables, pareto_front_objectives, pareto_mask, hv = (
            compute_hypervolume_and_pf(
                variable_data,
                objective_data,
                self.torch_reference_point,
            )
        )

        # scale the pareto front objectives back to original space
        if pareto_front_objectives is not None:
            pareto_front_objectives = pareto_front_objectives / weights

        return (
            pareto_front_variables,
            pareto_front_objectives,
            pareto_mask,
            hv,
        )

    def update_pareto_front_history(self):
        """
        Update the historical pareto front statistics in the generator.

        For each row of data in self.data, compute the pareto front stats
        (hypervolume, number of non-dominated points) if there is no
        corresponding entry exists in the `self.pareto_front_history` DataFrame.
        """
        # TODO: make sure this works when manually changing the data frame
        if self.pareto_front_history is None:
            self.pareto_front_history = pd.DataFrame()

        # for each row of data, compute the cumulative pareto front stats
        for i in self.data.index:
            # check if the pareto front stats already exist
            if i in self.pareto_front_history.index:
                continue

            # get scaled data
            variable_data, objective_data, _ = self._get_scaled_data(
                data=self.data.loc[:i]
            )

            # compute the pareto front stats
            _, pareto_front_variables, _, hv = compute_hypervolume_and_pf(
                variable_data,
                objective_data,
                self.torch_reference_point,
            )

            # get the number of non-dominated points
            n_non_dominated = (
                len(pareto_front_variables) if pareto_front_variables is not None else 0
            )

            # create a new row for the pareto front stats
            new_row: dict[str, Any] = {
                "iteration": i,
                "hypervolume": hv,
                "n_non_dominated": n_non_dominated,
            }
            # add the new row to the pareto front history
            self.pareto_front_history = pd.concat(
                [
                    self.pareto_front_history,
                    pd.DataFrame(new_row, index=[i]),
                ],
                ignore_index=False,
            )

    def _get_scaled_data(self, data: pd.DataFrame):
        """get scaled input/objective data for use with botorch logic which assumes
        maximization for each objective"""

        # get raw data
        var_df, obj_df, _, _ = self.vocs.extract_data(
            data, return_raw=True, return_valid=True
        )

        variable_data = torch.tensor(var_df[self.vocs.variable_names].to_numpy())
        objective_data = torch.tensor(obj_df[self.vocs.objective_names].to_numpy())
        weights = set_botorch_weights(self.vocs)[: self.vocs.n_objectives]
        return variable_data, objective_data * weights, weights


def formatted_base_docstring():
    doc = BayesianGenerator.__doc__ or ""
    return "\nBase Generator\n---------------\n" + doc
