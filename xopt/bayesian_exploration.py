import logging
from functools import partial

from botorch.acquisition import GenericMCObjective
from botorch.optim.optimize import optimize_acqf

from .bayesian.acquisition.exploration import qBayesianExploration, BayesianExploration
from .bayesian.optimize import bayesian_optimize

"""
    Bayesian Exploration Botorch

"""

# Logger
logger = logging.getLogger(__name__)


def bayes_exp_acq(model,
                  bounds,
                  vocs,
                  sigma=None,
                  sampler=None,
                  batch_size=1,
                  num_restarts=20,
                  raw_samples=1024):
    """

    Optimize Bayesian Exploration

    model should be a SingleTaskGP model trained such that the output has a shape n x m + 1
    where the first element is the target function for exploration and m is the number of constraints

    """

    n_constraints = len(vocs['constraints'])

    # serialized Bayesian Exploration
    if batch_size == 1:
        constraint_dict = {}
        for i in range(1, n_constraints + 1):
            constraint_dict[i] = [None, 0.0]

        acq_func = BayesianExploration(model, 0, constraint_dict, sigma)

    # batched Bayesian Exploration
    else:
        assert sigma is None  # proximal biasing not possible in batched context

        mc_obj = GenericMCObjective(lambda Z, X: Z[..., 0])

        # define constraint functions - note issues with lambda implementation
        # https://tinyurl.com/j8wmckd3
        def constr_func(Z, index=-1):
            return Z[..., index]

        constraint_functions = []
        for i in range(1, n_constraints + 1):
            constraint_functions += [partial(constr_func, index=-i)]

        acq_func = qBayesianExploration(model, sampler, mc_obj, constraints=constraint_functions)

    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,  # used for initialization heuristic
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=True,
    )

    return candidates.detach()


def bayesian_exploration(config, evaluate_f, **kwargs):
    return bayesian_optimize(config, evaluate_f, bayes_exp_acq, **kwargs)
