import logging
import math
from abc import ABC
from dataclasses import dataclass
from typing import Dict, List

import torch
from botorch.optim import optimize_acqf

from xopt.generators.bayesian.bayesian_generator import BayesianGenerator

from xopt.generators.bayesian.options import BayesianOptions
from xopt.vocs import VOCS

logger = logging.getLogger()


"""
Functions and classes that support TuRBO - an algorithm that fits a collection of
local models and
performs a principled global allocation of samples across these models via an
implicit bandit approach
https://proceedings.neurips.cc/paper/2019/file/6c990b7aca7bc7058f5e98ea909e924b-Paper.pdf
"""


class TuRBOBayesianGenerator(BayesianGenerator, ABC):
    def __init__(self, vocs: VOCS, options: BayesianOptions = None):
        super(TuRBOBayesianGenerator, self).__init__(vocs, options)

        self.turbo_state = None
        self.reset_turbo_state()

    def generate(self, n_candidates: int) -> List[Dict]:
        # if no data exists use random generator to generate candidates
        if self.data.empty:
            return self.vocs.random_inputs(self.options.n_initial)
        else:
            # update the turbo state with the most recent observation
            objective_data = self.vocs.objective_data(self.data, "")
            y_last = torch.tensor(objective_data.iloc[-1].to_numpy(), **self._tkwargs)
            update_state(self.turbo_state, y_last)

            # train model
            self.train_model(self.data)

            # get trust region
            trust_region = self.get_trust_region()

            # get acquisition function
            acq_funct = self.get_acquisition(self._model)

            # get initial points for optimization (if specified)
            batch_initial_points, raw_samples = self._get_initial_batch_points(
                trust_region
            )

            # optimize acquisition function inside trust region
            candidates, out = optimize_acqf(
                acq_function=acq_funct,
                bounds=trust_region,
                q=n_candidates,
                raw_samples=raw_samples,
                batch_initial_conditions=batch_initial_points,
                num_restarts=self.options.optim.num_restarts,
            )
            logger.debug("Best candidate from optimize", candidates, out)
            return self.vocs.convert_numpy_to_inputs(candidates.detach().cpu().numpy())

    def get_trust_region(self):
        if self.model is None:
            raise RuntimeError("getting trust region requires a GP model to be trained")

        bounds = self._get_bounds()

        # get location of best point so far
        variable_data = self.vocs.variable_data(self.data, "")
        objective_data = self.vocs.objective_data(self.data, "")
        # best_f = torch.tensor(objective_data.max(), **self._tkwargs)
        # note that the trust region will be around the minimum point since Xopt
        # assumes minimization
        best_idx = objective_data.idxmin()
        best_x = torch.tensor(
            variable_data.loc[best_idx][self.vocs.variable_names].to_numpy(),
            **self._tkwargs
        )

        # Scale the TR to be proportional to the lengthscales of the objective model
        x_center = best_x.clone()
        weights = self.model.models[0].covar_module.base_kernel.lengthscale.detach()
        # weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * self.turbo_state.length / 2.0, *bounds)
        tr_ub = torch.clamp(x_center + weights * self.turbo_state.length / 2.0, *bounds)
        return torch.cat((tr_lb, tr_ub), dim=0)

    def reset_turbo_state(self):
        self.turbo_state = TurboState(self.vocs.n_variables, 1)


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 1.0
    length_min: float = 0.5
    length_max: float = 100
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 2  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state, Y_next):
    """
    update turbo state class
    NOTE: this is the opposite of botorch which assumes maximization, xopt assumes
    minimization
    """
    if max(Y_next) < state.best_value + 1e-3 * math.fabs(state.best_value):
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

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state
