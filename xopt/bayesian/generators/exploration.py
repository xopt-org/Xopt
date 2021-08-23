import logging
from functools import partial

import torch
from botorch.acquisition import GenericMCObjective
from botorch.optim.optimize import optimize_acqf

from ..acquisition.exploration import qBayesianExploration, BayesianExploration
from ..utils import get_bounds
#Logger
logger = logging.getLogger(__name__)


class BayesianExplorationGenerator:
    def __init__(self, vocs,
                 batch_size=1,
                 sigma=None,
                 sampler=None,
                 num_restarts=20,
                 raw_samples=1024):

        self.vocs = vocs
        self.batch_size = batch_size
        self.sigma = sigma
        self.sampler = sampler
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.acq_func = BayesianExploration

    def generate(self, model, **tkwargs):
        """

        Optimize Bayesian Exploration

        model should be a SingleTaskGP model trained such that the output has a shape n x m + 1
        where the first element is the target function for exploration and m is the number of constraints

        """
        n_constraints = len(self.vocs['constraints'])
        n_variables = len(self.vocs['variables'])
        bounds = get_bounds(self.vocs, **tkwargs)

        # serialized Bayesian Exploration
        if self.batch_size == 1:
            if self.sigma is None:
                self.sigma = torch.eye(n_variables, **tkwargs) * 1e10

            elif not isinstance(self.sigma, torch.Tensor):
                self.sigma = torch.tensor(self.sigma.copy(), **tkwargs)

            constraint_dict = {}
            for i in range(1, n_constraints + 1):
                constraint_dict[i] = [None, 0.0]

            constraint_dict = constraint_dict if len(constraint_dict) else None
            acq_func = self.acq_func(model, 0, constraint_dict, self.sigma)

        # batched Bayesian Exploration
        else:
            assert self.sigma is None, 'proximal biasing not possible in batched context'

            mc_obj = GenericMCObjective(lambda Z, X: Z[..., 0])

            # define constraint functions - note issues with lambda implementation
            # https://tinyurl.com/j8wmckd3
            def constr_func(Z, index=-1):
                return Z[..., index]

            constraint_functions = []
            for i in range(1, n_constraints + 1):
                constraint_functions += [partial(constr_func, index=-i)]

            acq_func = qBayesianExploration(model, self.sampler, mc_obj, constraints=constraint_functions)

        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=self.batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,  # used for initialization heuristic
            options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
            sequential=True,
        )

        return candidates.detach()

