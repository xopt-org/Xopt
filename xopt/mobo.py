import logging
import os
import random
import sys
import traceback
import time

from functools import partial

import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.fit import fit_gpytorch_model
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.transforms import standardize
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from xopt.tools import full_path, DummyExecutor

"""
    Multi-objective Bayesian optimization (MOBO) using EHVI and Botorch

"""

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# Logger
logger = logging.getLogger(__name__)


class NoValidResultsError(Exception):
    pass


def sampler_evaluate(inputs, evaluate_f, verbose=False):
    """
    Wrapper to catch any exceptions


    inputs: possible inputs to evaluate_f (a single positional argument)

    evaluate_f: a function that takes a dict with keys, and returns some output

    """
    global outputs
    result = {}

    try:
        outputs = evaluate_f(inputs)
        err = False

    except Exception as ex:
        outputs = {'Exception': str(ex),
                   'Traceback': traceback.print_tb(ex.__traceback__)}
        err = True
        if verbose:
            print(outputs)

    finally:
        result['inputs'] = inputs
        result['outputs'] = outputs
        result['error'] = err

    return result


def get_results(futures):
    # check the status of all futures
    results = []
    done = False
    ii = 1
    n_samples = len(futures)
    while not done:
        for ix in range(n_samples):
            fut = futures[ix]

            if not fut.done():
                continue

            results.append(fut.result())
            ii += 1

        if ii > n_samples:
            done = True

        # Slow down polling. Needed for MPI to work well.
        time.sleep(0.001)

    return results


def collect_results(results, vocs):
    """
        Collect successful measurement results into torch tensors to add to training data
    """

    train_x = []
    train_y = []
    train_c = []

    at_least_one_point = False

    for result in results:
        if not result['error']:
            train_x += [[result['inputs'][ele] for ele in vocs['variables'].keys()]]
            train_y += [[result['outputs'][ele] for ele in vocs['objectives'].keys()]]
            train_c += [[result['outputs'][ele] for ele in vocs['constraints'].keys()]]

            at_least_one_point = True

    if not at_least_one_point:
        raise NoValidResultsError('No valid results')

    train_x = torch.tensor(train_x, **tkwargs)
    train_y = torch.tensor(train_y, **tkwargs)
    train_c = torch.tensor(train_c, **tkwargs)

    return train_x, train_y, train_c


def plot_acq(model, bounds, ref_point, partitioning, sampler, n_obectives, n_constraints, constraint_functions):
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
        sampler=sampler,
        # define an objective that specifies which outcomes are the objectives
        objective=IdentityMCMultiOutputObjective(outcomes=list(range(n_obectives))),
    )

    constr_acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),  # use known reference point
        partitioning=partitioning,
        sampler=sampler,
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


def optimize_qehvi(model,
                   train_y,
                   train_c,
                   bounds,
                   ref_point,
                   sampler,
                   batch_size=1,
                   num_restarts=20,
                   raw_samples=1024,
                   plot=False):
    """Optimizes the qEHVI acquisition function and returns new candidate(s)."""
    n_obectives = train_y.shape[-1]
    n_constraints = train_c.shape[-1]

    # compute feasible observations
    is_feas = (train_c <= 0).all(dim=-1)
    print(f'n_feas: {torch.count_nonzero(is_feas)}')
    # compute points that are better than the known reference point
    better_than_ref = (train_y > ref_point).all(dim=-1)
    # partition non-dominated space into disjoint rectangles
    partitioning = NondominatedPartitioning(
        ref_point=ref_point,
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
        ref_point=ref_point.tolist(),  # use known reference point
        partitioning=partitioning,
        sampler=sampler,
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
                 ref_point, partitioning,
                 sampler, n_obectives, n_constraints,
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


def get_corrected_outputs(vocs, ref, train_y, train_c):
    """
    scale and invert outputs depending on maximization/minimization, etc.
    """

    objectives = vocs['objectives']
    objective_names = list(objectives.keys())

    constraints = vocs['constraints']
    constraint_names = list(constraints.keys())

    # need to multiply -1 for each axis that we are using 'MINIMIZE' for an objective
    # need to multiply -1 for each axis that we are using 'GREATER_THAN' for a constraint
    corrected_ref = ref.clone()
    corrected_train_y = train_y.clone()
    corrected_train_c = train_c.clone()

    # negate objective measurements that want to be minimized
    for j, name in zip(range(len(objective_names)), objective_names):
        if vocs['objectives'][name] == 'MINIMIZE':
            corrected_train_y[:, j] = -train_y[:, j]
            corrected_ref[j] = -ref[j]

        elif vocs['objectives'][name] == 'MAXIMIZE':
            pass
        else:
            logger.warning(f'Objective goal {vocs["objectives"][name]} not found, defaulting to MAXIMIZE')

    # negate constraints that use 'GREATER_THAN'
    for k, name in zip(range(len(constraint_names)), constraint_names):
        if vocs['constraints'][name][0] == 'GREATER_THAN':
            corrected_train_c[:, k] = (vocs['constraints'][name][1] - train_c[:, k])

        elif vocs['constraints'][name][0] == 'LESS_THAN':
            corrected_train_c[:, k] = -(vocs['constraints'][name][1] - train_c[:, k])
        else:
            logger.warning(f'Constraint goal {vocs["constraints"][name]} not found, defaulting to LESS_THAN')

    return corrected_train_y, corrected_train_c, corrected_ref


def mobo(vocs, evaluate_f, ref,
         n_steps=30,
         mc_samples=128,
         batch_size=1,
         executor=None,
         n_initial_samples=5,
         model_options=None,
         seed=None,
         output_path=None,
         verbose=True,
         return_model=False,
         initial_x=None,
         plot_acq=False,
         use_gpu=False):
    """

    Parameters
    ----------
    vocs : dict
        Varabiles, objectives, constraints and statics dictionary, see xopt documentation for detials

    evaluate_f : callable
        Returns dict of outputs after problem has been evaluated

    ref : torch.Tensor
        Reference point for multi-objective optimization

    n_steps : int, default: 30
        Number of optimization steps to take

    mc_samples : int, default: 128
        Number of monte carlo samples to take when computing the acquisition function intergral

    batch_size : int, default: 1
        Number of candidates to generate at each optimization step

    executor : futures.Executor, default: None
        Executor object to evaluate problem using multiple threads or processors

    n_initial_samples : int, defualt: 5
        Number of initial sobel_random samples to take to start optimization. Ignored if initial_x is not None.

    model_options : dict, optional
        Arguments to Gaussian Process model creation, ie. custom kernels, likelihoods

    seed : int, optional
        Seed for random number generation to freeze random number generation.

    output_path : str, optional
        Location to save optimization data and models.

    verbose : bool, default: False
        Specify if the algorithm should print optimization progress.

    return_model : bool, default: False
        Specify if the algorithm should return the final trained model

    initial_x : torch.Tensor, optional
        Initial input points to sample evaluator at, causes sampling to ignore n_initial_samples

    plot_acq : bool, False
        Specify if the algorithm should plot the GP predictions and acquisition function values in
        the input domain at the end of optimization.

    use_gpu : bool, False
        Specify if the algorithm should use GPU resources if available. Only use on large problems!

    Returns
    -------
    train_x : torch.Tensor
        Observed variable values

    train_y : torch.Tensor
        Observed objective values

    train_c : torch.Tensor
        Observed constraint values

    model : SingleTaskGP
        If return_model = True this contains the final trained model for objectives and constraints.

    """

    random.seed(seed)
    if model_options is None:
        model_options = {}

    if not use_gpu:
        tkwargs['device'] = 'cpu'

    # convert ref tensor to match model device
    ref = ref.to(**tkwargs)

    # Verbose print helper
    def vprint(*a, **k):
        # logger.debug(' '.join(a))
        # TODO: use logging instead of print statements
        if verbose:
            print(*a, **k)
            sys.stdout.flush()

    if not executor:
        serial = True
        executor = DummyExecutor()
        vprint('No executor given. Running in serial mode.')

    # Setup saving to file
    if output_path:
        path = full_path(output_path)
        assert os.path.exists(path), f'output_path does not exist {path}'

        def save(pop, prefix, generation):
            # TODO: implement this
            raise NotImplementedError

    else:
        # Dummy save
        def save(pop, prefix, generation):
            pass

    # parse VOCS
    variables = vocs['variables']
    variable_names = list(variables.keys())

    objectives = vocs['objectives']
    objective_names = list(objectives.keys())

    constraints = vocs['constraints']
    constraint_names = list(constraints.keys())

    # get initial bounds
    bounds = torch.transpose(
        torch.vstack([torch.tensor(ele, **tkwargs) for _, ele in variables.items()]), 0, 1)

    # create normalization transforms for model inputs
    # inputs are normalized in [0,1]
    input_normalize = Normalize(len(variable_names), bounds)

    # generate initial samples if no initial samples are given
    if initial_x is None:
        initial_x = draw_sobol_samples(bounds, 1, n_initial_samples)[0]

    # submit evaluation of initial samples
    sampler_evaluate_args = {'verbose': verbose}
    initial_y = [executor.submit(sampler_evaluate,
                                 dict(zip(variable_names, x.cpu().numpy())),
                                 evaluate_f,
                                 **sampler_evaluate_args) for x in initial_x]
    results = get_results(initial_y)
    train_x, train_y, train_c = collect_results(results, vocs)

    hv_track = []

    # do optimization
    for i in range(n_steps):

        # get corrected values
        corrected_train_y, corrected_train_c, corrected_ref = get_corrected_outputs(vocs, ref, train_y, train_c)

        # standardize y training data
        standardized_train_y = standardize(corrected_train_y)

        # horiz. stack objective and constraint results for training/acq specification
        train_outputs = torch.hstack((standardized_train_y, corrected_train_c))

        model = SingleTaskGP(train_x, train_outputs,
                             input_transform=input_normalize,
                             **model_options)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        # create MC sampler
        mc_sampler = SobolQMCNormalSampler(mc_samples)

        # get candidate point(s)
        candidates = optimize_qehvi(model,
                                    standardized_train_y,
                                    corrected_train_c,
                                    bounds,
                                    corrected_ref,
                                    mc_sampler,
                                    batch_size=batch_size)

        if verbose:
            vprint(candidates)

        # observe candidates
        fut = [executor.submit(sampler_evaluate,
                               dict(zip(variable_names, x.cpu().numpy())),
                               evaluate_f,
                               **sampler_evaluate_args) for x in candidates]
        results = get_results(fut)

        try:
            new_x, new_y, new_c = collect_results(results, vocs)

            # add new observations to training data
            train_x = torch.vstack((train_x, new_x))
            train_y = torch.vstack((train_y, new_y))
            train_c = torch.vstack((train_c, new_c))
        except NoValidResultsError:
            print('No valid results found, skipping to next iteration')
            continue

        # give some feedback to the user
        is_feas = (corrected_train_c <= 0).all(dim=-1)
        feas_train_obj = corrected_train_y[is_feas]
        if feas_train_obj.shape[0] > 0:
            # define hypervolume calculator
            hv = Hypervolume(ref_point=corrected_ref)

            pareto_mask = is_non_dominated(feas_train_obj)
            pareto_y = feas_train_obj[pareto_mask]
            # compute hypervolume - remember to
            volume = hv.compute(pareto_y)
        else:
            volume = 0.0
        hv_track += [volume]
        if verbose:
            print(f'Step : {i}, hypervolume : {volume}')

    if plot_acq:
        # get candidate point(s)
        candidates = optimize_qehvi(model,
                                    corrected_train_y,
                                    corrected_train_c,
                                    bounds,
                                    corrected_ref,
                                    mc_sampler,
                                    plot=True)

    if return_model:
        # get corrected values
        corrected_train_y, corrected_train_c, corrected_ref = get_corrected_outputs(vocs, ref, train_y, train_c)

        # horiz. stack objective and constraint results for training/acq specification
        train_outputs = torch.hstack((standardized_train_y, corrected_train_c))

        model = SingleTaskGP(train_x, train_outputs,
                             input_transform=input_normalize,
                             **model_options)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        return train_x, train_y, train_c, model

    else:
        return train_x.cpu(), train_y.cpu(), train_c.cpu()
