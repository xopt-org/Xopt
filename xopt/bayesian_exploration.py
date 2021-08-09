import logging
from functools import partial

import torch
from botorch.acquisition import GenericMCObjective
from botorch.optim.optimize import optimize_acqf

from .bayesian.acquisition.exploration import qBayesianExploration, BayesianExploration
from .bayesian.optimize import bayesian_optimize
from .bayesian.utils import check_config

"""
    Bayesian Exploration Botorch

"""

# Logger
logger = logging.getLogger(__name__)


def bayes_exp_acq(model,
                  bounds,
                  vocs,
                  options):
    """

    Optimize Bayesian Exploration

    model should be a SingleTaskGP model trained such that the output has a shape n x m + 1
    where the first element is the target function for exploration and m is the number of constraints

    """
    n_constraints = len(vocs['constraints'])
    batch_size = options.get('batch_size', 1)
    sigma = options.get('sigma', None)
    sampler = options.get('sampler', None)
    num_restarts = options.get('n_restarts', 20)
    raw_samples = options.get('raw_samples', 1024)

    if sigma is not None:
        sigma = torch.tensor(sigma.copy())

    # serialized Bayesian Exploration
    if batch_size == 1:
        constraint_dict = {}
        for i in range(1, n_constraints + 1):
            constraint_dict[i] = [None, 0.0]

        acq_func = BayesianExploration(model, 0, constraint_dict, sigma)

    # batched Bayesian Exploration
    else:
        assert sigma is None, 'proximal biasing not possible in batched context'

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
    """
        Bayesian exploration algorithm

        Parameters
        ----------
        config : dict
            Varabiles, objectives, constraints and statics dictionary, see xopt documentation for detials

        evaluate_f : callable
            Returns dict of outputs after problem has been evaluated


        Optional kwargs arguments
        --------
        n_steps : int, default: 30
            Number of optimization steps to take

        executor : futures.Executor, default: None
            Executor object to evaluate problem using multiple threads or processors

        n_initial_samples : int, defualt: 5
            Number of initial sobel_random samples to take to start optimization. Ignored if initial_x is not None.

        custom_model : callable, optional
            Function in the form f(train_x, train_y) that returns a botorch model instance

        output_path : str, optional
            Location to save optimization data and models.

        verbose : bool, default: False
            Specify if the algorithm should print optimization progress.

        restart_data_file : str, optional
            Pickled pandas data frame object containing initial data for restarting optimization.

        initial_x : torch.Tensor, optional
            Initial input points to sample evaluator at, causes sampling to ignore n_initial_samples

        use_gpu : bool, False
            Specify if the algorithm should use GPU resources if available. Only use on large problems!

        eval_args : list, []
            List of positional arguments for evaluation function

        Returns
        -------
        results : dict
            Dictionary object containing optimization points + other info

        """

    config = check_config(config, __name__, **kwargs)
    return bayesian_optimize(config, evaluate_f, bayes_exp_acq)
