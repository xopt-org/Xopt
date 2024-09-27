import logging
import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import pandas as pd
import torch
from pydantic import ConfigDict, Field, PositiveFloat, PositiveInt

from xopt.pydantic import XoptBaseModel
from xopt.vocs import VOCS

logger = logging.getLogger()

"""
Functions and classes that support TuRBO - an algorithm that fits a collection of
local models and
performs a principled global allocation of samples across these models via an
implicit bandit approach
https://proceedings.neurips.cc/paper/2019/file/6c990b7aca7bc7058f5e98ea909e924b-Paper.pdf
"""


class TurboController(XoptBaseModel, ABC):
    vocs: VOCS = Field(exclude=True)
    dim: PositiveInt
    batch_size: PositiveInt = Field(1, description="number of trust regions to use")
    length: float = Field(
        0.25,
        description="base length of trust region",
        ge=0.0,
    )
    length_min: PositiveFloat = 0.5**7
    length_max: PositiveFloat = Field(
        2.0,
        description="maximum base length of trust region",
    )
    failure_counter: int = Field(0, description="number of failures since reset", ge=0)
    failure_tolerance: PositiveInt = Field(
        None, description="number of failures to trigger a trust region expansion"
    )
    success_counter: int = Field(0, description="number of successes since reset", ge=0)
    success_tolerance: PositiveInt = Field(
        None,
        description="number of successes to trigger a trust region contraction",
    )
    center_x: Optional[Dict[str, float]] = Field(None)
    scale_factor: float = Field(
        2.0, description="multiplier to increase or decrease trust region", ge=1.0
    )
    restrict_model_data: Optional[bool] = Field(
        True, description="flag to restrict model data to within the trust region"
    )
    tkwargs: Dict[str, Union[torch.dtype, str]] = Field({"dtype": torch.double})

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    def __init__(self, vocs: VOCS, **kwargs):
        dim = vocs.n_variables

        super(TurboController, self).__init__(vocs=vocs, dim=dim, **kwargs)

        # initialize tolerances if not specified
        if self.failure_tolerance is None:
            self.failure_tolerance = int(
                math.ceil(
                    max(
                        [2.0 / self.batch_size, float(self.dim) / 2.0 * self.batch_size]
                    )
                )
            )

        if self.success_tolerance is None:
            self.success_tolerance = int(
                math.ceil(
                    max(
                        [2.0 / self.batch_size, float(self.dim) / 2.0 * self.batch_size]
                    )
                )
            )

    def get_trust_region(self, generator):
        model = generator.model
        bounds = torch.tensor(self.vocs.bounds, **self.tkwargs)

        if self.center_x is not None:
            # get bounds width
            bound_widths = bounds[1] - bounds[0]

            # Scale the TR to be proportional to the lengthscales of the objective model
            x_center = torch.tensor(
                [self.center_x[ele] for ele in self.vocs.variable_names], **self.tkwargs
            ).unsqueeze(dim=0)

            if model.models[0].covar_module.base_kernel.lengthscale is not None:
                lengthscales = model.models[
                    0
                ].covar_module.base_kernel.lengthscale.detach()

                # calculate the ratios of lengthscales for each axis
                weights = lengthscales / torch.prod(lengthscales) ** (1 / self.dim)
            else:
                weights = 1.0

            # calculate the tr bounding box
            tr_lb = torch.clamp(
                x_center - weights * self.length * bound_widths / 2.0, *bounds
            )
            tr_ub = torch.clamp(
                x_center + weights * self.length * bound_widths / 2.0, *bounds
            )
            return torch.cat((tr_lb, tr_ub), dim=0)
        else:
            return bounds

    def update_trust_region(self):
        if self.success_counter == self.success_tolerance:  # Expand trust region
            self.length = min(self.scale_factor * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:  # Shrink trust region
            self.length = max(self.length / self.scale_factor, self.length_min)
            self.failure_counter = 0

    def get_data_in_trust_region(self, data: pd.DataFrame, generator):
        """
        Get subset of data in trust region
        """
        variable_data = torch.tensor(self.vocs.variable_data(data).to_numpy())

        bounds = self.get_trust_region(generator)

        mask = torch.ones(len(variable_data), dtype=torch.bool)
        for dim in range(variable_data.shape[1]):
            mask &= (variable_data[:, dim] >= bounds[0][dim]) & (
                variable_data[:, dim] <= bounds[1][dim]
            )

        return data.iloc[mask.numpy()]

    @abstractmethod
    def update_state(self, generator, previous_batch_size: int = 1) -> None:
        pass


class OptimizeTurboController(TurboController):
    name: str = Field("optimize", frozen=True)
    best_value: Optional[float] = None

    @property
    def minimize(self) -> bool:
        return self.vocs.objectives[self.vocs.objective_names[0]] == "MINIMIZE"

    def _set_best_point_value(self, data):
        # get location of best point so far
        variable_data = self.vocs.variable_data(data, "")
        objective_data = self.vocs.objective_data(data, "", return_raw=True)

        if self.minimize:
            best_idx = objective_data.idxmin()
            self.best_value = objective_data.min()[self.vocs.objective_names[0]]
        else:
            best_idx = objective_data.idxmax()
            self.best_value = objective_data.max()[self.vocs.objective_names[0]]

        self.center_x = (
            variable_data.loc[best_idx][self.vocs.variable_names].iloc[0].to_dict()
        )

    def update_state(self, generator, previous_batch_size: int = 1) -> None:
        """
        Update turbo state class using min of data points that are feasible.
        If no points in the data set are feasible raise an error.

        NOTE: this is the opposite of botorch which assumes maximization, xopt assumes
        minimization

        Parameters
        ----------
        generator : BayesianGenerator
            Entire data set containing previous measurements. Requires at least one
            valid point.

        previous_batch_size : int, default = 1
            Number of candidates in previous batch evaluation

        Returns
        -------
            None

        """
        data = generator.data

        # get locations of valid data samples
        feas_data = self.vocs.feasibility_data(data)

        if len(data[feas_data["feasible"]]) == 0:
            raise RuntimeError(
                "turbo requires at least one valid point in the training dataset"
            )
        else:
            self._set_best_point_value(data[feas_data["feasible"]])

        # get feasibility of last `n_candidates`
        recent_data = data.iloc[-previous_batch_size:]
        f_data = self.vocs.feasibility_data(recent_data)
        recent_f_data = recent_data[f_data["feasible"]]

        # if none of the candidates are valid count this as a failure
        if len(recent_f_data) == 0:
            self.success_counter = 0
            self.failure_counter += 1

        else:
            # if we had previous feasible points we need to compare with previous
            # best values, NOTE: this is the opposite of botorch which assumes
            # maximization, xopt assumes minimization
            Y_last = recent_f_data[self.vocs.objective_names[0]].min()

            if Y_last < self.best_value + 1e-3 * math.fabs(self.best_value):
                self.success_counter += 1
                self.failure_counter = 0
            else:
                self.success_counter = 0
                self.failure_counter += 1

        self.update_trust_region()


class SafetyTurboController(TurboController):
    name: str = Field("safety", frozen=True)
    scale_factor: float = 1.25
    min_feasible_fraction: float = 0.75

    def update_state(self, generator, previous_batch_size: int = 1):
        data = generator.data

        # set center point to be mean of valid data points
        feas = data[self.vocs.feasibility_data(data)["feasible"]]
        self.center_x = feas[self.vocs.variable_names].mean().to_dict()

        # get the feasibility fractions of the last batch
        last_batch = self.vocs.feasibility_data(data).iloc[-previous_batch_size:]
        feas_fraction = last_batch["feasible"].sum() / len(last_batch)

        if feas_fraction > self.min_feasible_fraction:
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        self.update_trust_region()


class EntropyTurboController(TurboController):
    name: str = Field("entropy", frozen=True)
    _best_entropy: float = None

    def update_state(self, generator, previous_batch_size: int = 1) -> None:
        if generator.algorithm_results is not None:
            # check to make sure required keys are in algorithm results
            for ele in ["solution_center", "solution_entropy"]:
                if ele not in generator.algorithm_results:
                    raise RuntimeError(
                        f"algorithm must include `{ele}` in "
                        f"`algorithm_results` property to use "
                        f"EntropyTurboController"
                    )

            self.center_x = dict(
                zip(
                    self.vocs.variable_names,
                    generator.algorithm_results["solution_center"],
                )
            )
            entropy = generator.algorithm_results["solution_entropy"]

            if self._best_entropy is not None:
                if entropy < self._best_entropy:
                    self.success_counter += 1
                    self.failure_counter = 0
                    self._best_entropy = entropy
                else:
                    self.success_counter = 0
                    self.failure_counter += 1

                self.update_trust_region()
            else:
                self._best_entropy = entropy
