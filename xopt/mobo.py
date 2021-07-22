import logging
import os
import random
import sys
import traceback
import time

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
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from xopt.tools import full_path, DummyExecutor

"""
    Multi-objective Bayesian optimization (MOBO) using EHVI and Botorch

"""


class NoValidResultsError(Exception):
    pass


def sampler_evaluate(inputs, evaluate_f, verbose=False):
    """
    Wrapper to catch any exceptions


    inputs: possible inputs to evaluate_f (a single positional argument)

    evaluate_f: a function that takes a dict with keys, and returns some output

    """
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

    train_x = torch.tensor(train_x)
    train_y = torch.tensor(train_y)
    train_c = torch.tensor(train_c)

    return train_x, train_y, train_c


def optimize_qehvi(model,
                   train_y,
                   train_c,
                   bounds,
                   ref_point,
                   sampler,
                   batch_size=1,
                   num_restarts=20,
                   raw_samples=1024,
                   plot = False):
    """Optimizes the qEHVI acquisition function and returns new candidate(s)."""
    n_obectives = train_y.shape[-1]

    # compute feasible observations
    is_feas = (train_c <= 0).all(dim=-1)
    # compute points that are better than the known reference point
    better_than_ref = (train_y > ref_point).all(dim=-1)
    # partition non-dominated space into disjoint rectangles
    partitioning = NondominatedPartitioning(
        ref_point=ref_point,
        # use observations that are better than the specified reference point and feasible
        # Y=train_y[better_than_ref & is_feas],
        Y=train_y[better_than_ref],

    )
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),  # use known reference point
        partitioning=partitioning,
        sampler=sampler,
        # define an objective that specifies which outcomes are the objectives
        objective=IdentityMCMultiOutputObjective(outcomes=list(range(n_obectives))),
        # define constraint function - see botorch docs for info - I'm not sure how it works
        # constraints=[lambda Z: Z[..., -1]]
    )

    standard_bounds = torch.zeros(bounds.shape)
    standard_bounds[1] = 1

    # plot the acquisition function for debugging purposes
    if plot:
        # only works in 2D
        assert bounds.shape[-1] == 2

        n = 100
        x = np.linspace(0, 1, n)
        xx = np.meshgrid(x, x)
        pts = torch.tensor(np.vstack((ele.ravel() for ele in xx)).T).float()

        with torch.no_grad():
            pts = pts.reshape(n**2, 1, 2)
            ehvi = acq_func.forward(pts)

        print(ehvi.shape)
        fig, ax = plt.subplots()
        c = ax.pcolor(*xx, ehvi.reshape(n, n))
        fig.colorbar(c)

    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,  # used for initialization heuristic
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=True,
    )

    return unnormalize(candidates.detach(), bounds=bounds)


def mobo(vocs, evaluate_f, ref,
         n_steps=30,
         mc_samples=128,
         executor=None,
         n_initial_samples=5,
         model_options=None,
         seed=None,
         output_path=None,
         verbose=True,
         return_model=False):
    """
        Multi-objective Bayeisan optimization

        Works with executors instantiated from:
            concurrent.futures.ProcessPoolExecutor
            concurrent.futures.ThreadPoolExecutor
            mpi4py.futures.MPIPoolExecutor


    """

    random.seed(seed)
    if model_options is None:
        model_options = {}

    # Logger
    logger = logging.getLogger(__name__)

    # Verbose print helper
    def vprint(*a, **k):
        # logger.debug(' '.join(a))
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
            pass

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
        torch.vstack([torch.tensor(ele) for _, ele in variables.items()]), 0, 1)

    # create normalization transforms for model inputs and outputs
    # inputs are normalized in [0,1]
    # outputs are standardized (mean = 0, std = 1)
    input_normalize = Normalize(len(variable_names), bounds)
    output_normalize = Standardize(len(objective_names) + len(constraint_names))

    # generate initial samples
    initial_x = draw_sobol_samples(bounds, 1, n_initial_samples)[0]

    # submit evaluation of initial samples
    sampler_evaluate_args = {'verbose': verbose}

    initial_y = [executor.submit(sampler_evaluate,
                                 dict(zip(variable_names, x.numpy())),
                                 evaluate_f,
                                 **sampler_evaluate_args) for x in initial_x]

    results = get_results(initial_y)

    train_x, train_y, train_c = collect_results(results, vocs)

    hv_track = []

    # do optimization
    for i in range(n_steps):
        # train model
        # horiz. stack objective and constraint results for easier training/acq specification
        train_outputs = torch.hstack((train_y, train_c))

        # need to multiply -1 for each axis that we are using 'MINIMIZE'
        corrected_ref = ref.clone()
        corrected_train_y = train_y.clone()
        # negate objective measurements that want to be minimized
        for j, name in zip(range(len(objective_names)), objective_names):
            if vocs['objectives'][name] == 'MINIMIZE':
                train_outputs[:, j] = -train_outputs[:, j]
                corrected_train_y[:, j] = -train_y[:, j]
                corrected_ref[j] = -ref[j]

            elif vocs['objectives'][name] == 'MAXIMIZE':
                pass
            else:
                logger.warning(f'Objective goal {vocs["objectives"][name]} not found, defaulting to MAXIMIZE')

        model = SingleTaskGP(train_x, train_outputs,
                             input_transform=input_normalize,
                             outcome_transform=output_normalize,
                             **model_options)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        # create MC sampler
        mc_sampler = SobolQMCNormalSampler(mc_samples)

        # get candidate point(s)
        candidates = optimize_qehvi(model,
                                    corrected_train_y,
                                    train_c,
                                    bounds,
                                    corrected_ref,
                                    mc_sampler)

        if verbose:
            print(candidates)

        # observe candidates
        fut = [executor.submit(sampler_evaluate,
                               dict(zip(variable_names, x.numpy())),
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
        # is_feas = (train_c <= 0).all(dim=-1)
        feas_train_obj = -train_y  # [is_feas]
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


    if return_model:
        model = SingleTaskGP(train_x, train_y,
                             input_transform=input_normalize,
                             #outcome_transform=output_normalize,
                             **model_options)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        return train_x, train_y, train_c, model

    else:
        return train_x, train_y, train_c
