import logging
import os
import random
import sys
import time
import traceback
from functools import partial

import botorch.models.model
import torch
from botorch.acquisition import GenericMCObjective
from botorch.fit import fit_gpytorch_model
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.optim.optimize import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from xopt.bayesian.acquisition.exploration import qBayesianExploration, BayesianExploration
from xopt.bayesian.utils import standardize
from xopt.tools import full_path, DummyExecutor


"""
    Bayesian Exploration Botorch

"""

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# Logger
logger = logging.getLogger(__name__)


class NoValidResultsError(Exception):
    pass


def sampler_evaluate(inputs, evaluate_f, *eval_args, verbose=False):
    """
    Wrapper to catch any exceptions


    inputs: possible inputs to evaluate_f (a single positional argument)

    evaluate_f: a function that takes a dict with keys, and returns some output

    """
    outputs = None
    result = {}

    try:
        outputs = evaluate_f(inputs, *eval_args)
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


def optimize_acq(model,
                 bounds,
                 n_constraints,
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


def get_corrected_constraints(vocs, train_c):
    """
    scale and invert outputs depending on maximization/minimization, etc.
    """

    constraints = vocs['constraints']
    constraint_names = list(constraints.keys())

    # need to multiply -1 for each axis that we are using 'GREATER_THAN' for a constraint
    corrected_train_c = train_c.clone()

    # negate constraints that use 'GREATER_THAN'
    for k, name in zip(range(len(constraint_names)), constraint_names):
        if vocs['constraints'][name][0] == 'GREATER_THAN':
            corrected_train_c[:, k] = (vocs['constraints'][name][1] - train_c[:, k])

        elif vocs['constraints'][name][0] == 'LESS_THAN':
            corrected_train_c[:, k] = -(vocs['constraints'][name][1] - train_c[:, k])
        else:
            logger.warning(f'Constraint goal {vocs["constraints"][name]} not found, defaulting to LESS_THAN')

    return corrected_train_c


def create_model(train_x, train_outputs, input_normalize, custom_model=None):
    # create model
    if custom_model is None:
        model = SingleTaskGP(train_x, train_outputs,
                             input_transform=input_normalize)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

    else:
        model = custom_model(train_x, train_outputs,
                             input_transform = input_normalize)
        assert isinstance(model, botorch.models.model.Model)

    return model


def bayesian_exploration(vocs, evaluate_f,
                         n_steps=30,
                         mc_samples=128,
                         batch_size=1,
                         sigma=None,
                         executor=None,
                         n_initial_samples=5,
                         custom_model=None,
                         seed=None,
                         output_path=None,
                         verbose=True,
                         return_model=False,
                         initial_x=None,
                         use_gpu=False,
                         eval_args=None):
    """

    Parameters
    ----------
    vocs : dict
        Varabiles, objectives, constraints and statics dictionary, see xopt documentation for detials

    evaluate_f : callable
        Returns dict of outputs after problem has been evaluated

    n_steps : int, default: 30
        Number of optimization steps to take

    mc_samples : int, default: 128
        Number of monte carlo samples to take when computing the acquisition function intergral

    batch_size : int, default: 1
        Number of candidates to generate at each optimization step

    sigma : torch.Tensor, optional
        Covariance matrix used for proximal biasing

    executor : futures.Executor, default: None
        Executor object to evaluate problem using multiple threads or processors

    n_initial_samples : int, defualt: 5
        Number of initial sobel_random samples to take to start optimization. Ignored if initial_x is not None.

    custom_model : callable, optional
        Function in the form f(train_x, train_y) that returns a botorch model instance

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

    use_gpu : bool, False
        Specify if the algorithm should use GPU resources if available. Only use on large problems!

    eval_args : list, []
        List of positional arguments for evaluation function

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

    if eval_args is None:
        eval_args = []
    random.seed(seed)

    if not use_gpu:
        tkwargs['device'] = 'cpu'

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
    assert len(objective_names) == 1

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
                                 evaluate_f, *eval_args,
                                 **sampler_evaluate_args) for x in initial_x]
    results = get_results(initial_y)
    train_x, train_y, train_c = collect_results(results, vocs)

    hv_track = []

    # do optimization
    for i in range(n_steps):

        # get corrected values
        corrected_train_c = get_corrected_constraints(vocs, train_c)

        # standardize y training data - use xopt version to allow for nans
        standardized_train_y = standardize(train_y)

        # horiz. stack objective and constraint results for training/acq specification
        train_outputs = torch.hstack((standardized_train_y, corrected_train_c))

        model = create_model(train_x, train_outputs, input_normalize, custom_model)

        # get candidate point(s)
        candidates = optimize_acq(model,
                                  bounds,
                                  len(constraint_names),
                                  batch_size=batch_size,
                                  sigma=sigma)

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

    if return_model:
        # get corrected values
        corrected_train_c = get_corrected_constraints(vocs, train_y, train_c)

        # horiz. stack objective and constraint results for training/acq specification
        train_outputs = torch.hstack((standardized_train_y, corrected_train_c))
        model = create_model(train_x, train_outputs, input_normalize, custom_model)

        return train_x, train_y, train_c, model

    else:
        return train_x.cpu(), train_y.cpu(), train_c.cpu()
