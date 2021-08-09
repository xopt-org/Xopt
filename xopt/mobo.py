import logging
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning

from .bayesian.utils import check_config
from .bayesian.optimize import bayesian_optimize

"""
    Multi-objective Bayesian optimization (MOBO) using EHVI and Botorch

"""

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# Logger
logger = logging.getLogger(__name__)


def plot_acq(model, bounds, ref_point, partitioning, n_obectives, n_constraints, constraint_functions):
    # only works in 2D
    assert bounds.shape[-1] == 2

    n = 50
    x = np.linspace(0, 3.14, n)
    xx = np.meshgrid(x, x)
    pts = torch.tensor(np.vstack((ele.ravel() for ele in xx)).T, **tkwargs)
    pts = pts.reshape(n ** 2, 1, 2)

    ehvi_acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),  # use known reference point
        partitioning=partitioning,
        # define an objective that specifies which outcomes are the objectives
        objective=IdentityMCMultiOutputObjective(outcomes=list(range(n_obectives))),
    )

    constr_acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),  # use known reference point
        partitioning=partitioning,
        # define an objective that specifies which outcomes are the objectives
        objective=IdentityMCMultiOutputObjective(outcomes=list(range(n_obectives))),
        # define constraint functions - a list of functions that map input b x q x m to b x q for each constraint
        constraints=constraint_functions
    )

    with torch.no_grad():
        ehvi = ehvi_acq_func(pts)

    fig, ax = plt.subplots()
    c = ax.pcolor(*xx, ehvi.reshape(n, n))
    fig.colorbar(c)

    with torch.no_grad():
        cehvi = constr_acq_func(pts)

    fig, ax = plt.subplots()
    c = ax.pcolor(*xx, cehvi.reshape(n, n))
    fig.colorbar(c)

    with torch.no_grad():
        pos = model.posterior(pts.squeeze())
        mean = pos.mean

    for j in range(n_obectives + n_constraints):
        fig, ax = plt.subplots()
        c = ax.pcolor(*xx, mean[:, j].reshape(n, n))
        ax.contour(xx[0].reshape(n, n),
                   xx[1].reshape(n, n),
                   mean[:, j].reshape(n, n), levels=[0.0])

        fig.colorbar(c)


def opt_mobo(model,
             bounds,
             vocs,
             ref=None,
             batch_size=1,
             num_restarts=20,
             raw_samples=1024,
             plot=False):

    """Optimizes the qEHVI acquisition function and returns new candidate(s)."""
    n_obectives = len(vocs['objectives'])
    n_constraints = len(vocs['constraints'])

    train_outputs = model.train_targets.T
    train_y = train_outputs[:, :n_obectives]
    train_c = train_outputs[:, n_obectives:]

    assert ref is not None

    # compute feasible observations
    is_feas = (train_c <= 0).all(dim=-1)
    print(f'n_feas: {torch.count_nonzero(is_feas)}')
    # compute points that are better than the known reference point
    better_than_ref = (train_y > ref).all(dim=-1)
    # partition non-dominated space into disjoint rectangles
    partitioning = NondominatedPartitioning(
        ref_point=ref,
        # use observations that are better than the specified reference point and feasible
        Y=train_y[better_than_ref & is_feas],

    )

    # define constraint functions - note issues with lambda implementation
    # https://tinyurl.com/j8wmckd3
    def constr_func(Z, index=-1):
        return Z[..., index]

    constraint_functions = []
    for i in range(1, n_constraints + 1):
        constraint_functions += [partial(constr_func, index=-i)]

    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref.tolist(),  # use known reference point
        partitioning=partitioning,
        # define an objective that specifies which outcomes are the objectives
        objective=IdentityMCMultiOutputObjective(outcomes=list(range(n_obectives))),
        # define constraint function - see botorch docs for info - I'm not sure how it works
        constraints=constraint_functions
    )

    standard_bounds = torch.zeros(bounds.shape)
    standard_bounds[1] = 1

    # plot the acquisition function and each model for debugging purposes
    if plot:
        plot_acq(model, bounds,
                 ref, partitioning,
                 n_obectives, n_constraints,
                 constraint_functions)

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

    # return unnormalize(candidates.detach(), bounds=bounds)
    return candidates.detach()


def get_corrected_ref(vocs, ref):
    ref = torch.tensor(ref)
    for j, name in zip(range(len(vocs['objectives'])), vocs['objectives'].keys()):
        if vocs['objectives'][name] == 'MINIMIZE':
            ref[j] = -ref[j]
    return ref


def mobo(config, evaluate_f, ref = None, **kwargs):
    """
          Constrained Multi-objective Bayesian Optimization

          Parameters
          ----------
          config : dict
              Varabiles, objectives, constraints and statics dictionary, see xopt documentation for detials

          evaluate_f : callable
              Returns dict of outputs after problem has been evaluated

          ref : list
              Reference point

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

    config, new_kwargs = check_config(config, __name__, **kwargs)

    # set reference point
    if ref is not None:
        new_kwargs['ref'] = get_corrected_ref(config['vocs'], ref)
    else:
        new_kwargs['ref'] = get_corrected_ref(config['vocs'],
                                              config['algorithm']['options']['ref'])

    return bayesian_optimize(config, evaluate_f, opt_mobo, **new_kwargs)
