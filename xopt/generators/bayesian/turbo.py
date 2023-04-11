import logging
import math
from dataclasses import dataclass

import torch

logger = logging.getLogger()


"""
Functions and classes that support TuRBO - an algorithm that fits a collection of
local models and
performs a principled global allocation of samples across these models via an
implicit bandit approach
https://proceedings.neurips.cc/paper/2019/file/6c990b7aca7bc7058f5e98ea909e924b-Paper.pdf
"""


def get_trust_region(vocs, model, bounds, data, turbo_state, tkwargs):
    if model is None:
        raise RuntimeError("getting trust region requires a GP model to be trained")

    # get bounds width
    bound_widths = bounds[1] - bounds[0]

    # get location of best point so far
    variable_data = vocs.variable_data(data, "")
    objective_data = vocs.objective_data(data, "")

    # note that the trust region will be around the minimum point since Xopt
    # assumes minimization
    best_idx = objective_data.idxmin()
    best_x = torch.tensor(
        variable_data.loc[best_idx][vocs.variable_names].to_numpy(), **tkwargs
    )

    # Scale the TR to be proportional to the lengthscales of the objective model
    x_center = best_x.clone()
    lengthscales = model.models[0].covar_module.base_kernel.lengthscale.detach()

    # calculate the ratios of lengthscales for each axis
    weights = lengthscales / torch.prod(lengthscales.pow(1.0 / len(lengthscales)))

    # calculate the tr bounding box
    tr_lb = torch.clamp(
        x_center - weights * turbo_state.length * bound_widths / 2.0, *bounds
    )
    tr_ub = torch.clamp(
        x_center + weights * turbo_state.length * bound_widths / 2.0, *bounds
    )
    return torch.cat((tr_lb, tr_ub), dim=0)


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.5  # normalized between zero and one
    length_min: float = 0.5**7
    length_max: float = 2
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([2.0 / self.batch_size, float(self.dim) / (self.batch_size)])
        )

        self.success_tolerance = math.ceil(
            max([2.0 / self.batch_size, float(self.dim) / (self.batch_size)])
        )


def update_state(state, Y_next):
    """
    update turbo state class
    NOTE: this is the opposite of botorch which assumes maximization, xopt assumes
    minimization
    """
    if min(Y_next) < state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = min(state.best_value, min(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state
