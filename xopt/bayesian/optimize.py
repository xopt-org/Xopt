import logging
import os
import sys
import time

import torch
from botorch.models.transforms import Standardize
from botorch.models.transforms.input import Normalize
from botorch.utils.sampling import draw_sobol_samples

from .data import save_data_dict, get_data_json
from .models.models import create_model
from .utils import standardize, collect_results, sampler_evaluate, get_corrected_outputs, NoValidResultsError
from ..tools import full_path, DummyExecutor

"""
    Bayesian Exploration Botorch

"""

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}

# Logger
logger = logging.getLogger(__name__)


def bayesian_optimize(vocs,
                      evaluate_f,
                      candidate_generator,
                      n_steps,
                      n_initial_samples,
                      output_path,
                      custom_model,
                      executor,
                      restart_file,
                      initial_x,
                      verbose,
                      use_gpu,
                      ):
    """
    Backend function for model based optimization

    Parameters
    ----------
    vocs : dict
        Varabiles, objectives, constraints and statics dictionary, see xopt documentation for detials

    evaluate_f : callable
        Returns dict of outputs after problem has been evaluated

    candidate_generator : object
        Generator object that has a generate(model, bounds, vocs, **tkwargs) method

    n_steps : int
        Number of optimization steps to execute

    n_initial_samples : int
        Number of initial samples to take before using the model, overwritten by initial_x

    output_path : str
        Path location to place outputs

    custom_model : callable
        Function of the form f(train_inputs, train_outputs) that returns a trained custom model

    executor : Executor
        Executor object to run evaluate_f

    restart_file : str
        File location of JSON file that has previous data

    initial_x : list
        Nested list to provide initial candiates to evaluate, overwrites n_initial_samples

    verbose : bool
        Print out messages during optimization

    use_gpu : bool
        Specify if GPU should be used if available

    Returns
    -------
    results : dict
        Dictionary with output data at the end of optimization
    """

    # Verbose print helper
    def vprint(*a, **k):
        # logger.debug(' '.join(a))
        # TODO: use logging instead of print statements
        if verbose:
            print(*a, **k)
            sys.stdout.flush()

    vprint(f'started running optimization with generator: {candidate_generator}')

    # set up gpu if requested
    if use_gpu:
        if torch.cuda.is_available():
            tkwargs['device'] = torch.device('cuda')
            vprint(f'using gpu device {torch.cuda.get_device_name(tkwargs["device"])}')
        else:
            logger.warning('gpu requested but not found, using cpu')

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

    # set executor
    exe = DummyExecutor() if executor is None else executor

    # parse VOCS
    variables = vocs['variables']
    variable_names = list(variables.keys())

    # get initial bounds
    bounds = torch.vstack([torch.tensor(ele, **tkwargs) for _, ele in variables.items()]).T

    # create normalization transforms for model inputs
    # inputs are normalized in [0,1]
    input_normalize = Normalize(len(variable_names), bounds)

    sampler_evaluate_args = {'verbose': verbose}

    # generate initial samples if no initial samples are given
    if restart_file is None:
        if initial_x is None:
            initial_x = draw_sobol_samples(bounds, 1, n_initial_samples)[0]
        else:
            initial_x = initial_x

        # submit evaluation of initial samples
        vprint('submitting initial candidates')
        initial_y = [exe.submit(sampler_evaluate,
                                     dict(zip(variable_names, x.cpu().numpy())),
                                     evaluate_f,
                                     **sampler_evaluate_args) for x in initial_x]

        train_x, train_y, train_c = collect_results(initial_y, vocs, **tkwargs)

    else:
        train_x, train_y, train_c = get_data_json(restart_file,
                                                  vocs, **tkwargs)

    check_training_data_shape(train_x, train_y, train_c, vocs)

    # do optimization
    vprint('starting optimization loop')
    for i in range(n_steps):
        check_training_data_shape(train_x, train_y, train_c, vocs)

        # get corrected values
        corrected_train_y, corrected_train_c = get_corrected_outputs(vocs, train_y, train_c)

        # standardize y training data - use xopt version to allow for nans
        standardized_train_y = standardize(corrected_train_y)

        # horiz. stack objective and constraint results for training/acq specification
        train_outputs = torch.hstack((standardized_train_y, corrected_train_c))

        # create and train model
        model_start = time.time()
        model = create_model(train_x, train_outputs, input_normalize, custom_model)
        vprint(f'Model creation time: {time.time() - model_start:.4} s')

        # get candidate point(s)
        candidate_start = time.time()
        candidates = candidate_generator.generate(model, bounds, vocs, **tkwargs)
        vprint(f'Candidate generation time: {time.time() - candidate_start:.4} s')
        vprint(f'Candidate(s): {candidates}')

        # observe candidates
        vprint('submitting candidates')
        fut = [exe.submit(sampler_evaluate,
                               dict(zip(variable_names, x.cpu().numpy())),
                               evaluate_f,
                               **sampler_evaluate_args) for x in candidates]
        try:
            new_x, new_y, new_c = collect_results(fut, vocs, **tkwargs)

            # add new observations to training data
            train_x = torch.vstack((train_x, new_x))
            train_y = torch.vstack((train_y, new_y))
            train_c = torch.vstack((train_c, new_c))

            # get feasibility values
            feas, constraint_status = get_feasability_constraint_status(train_y, train_c, vocs)

            full_data = torch.hstack((train_x, train_y, train_c, constraint_status, feas))
            save_data_dict(vocs, full_data, output_path)

        except NoValidResultsError:
            print('No valid results found, skipping to next iteration')
            continue

    # horiz. stack objective and constraint results for training/acq specification
    train_outputs = torch.hstack((train_y, train_c))

    feas, constraint_status = get_feasability_constraint_status(train_y, train_c, vocs)

    # output transformer
    output_standardize = Standardize(train_outputs.shape[-1])
    model = create_model(train_x, train_outputs,
                         input_normalize, custom_model,
                         outcome_transform=output_standardize)

    results = {'variables': train_x.cpu(),
               'objectives': train_y.cpu(),
               'constraints': train_c.cpu(),
               'constraint_status': constraint_status.cpu(),
               'feasibility': feas.cpu(),
               'model': model.cpu()}

    return results


def get_feasability_constraint_status(train_y, train_c, vocs):
    _, corrected_train_c = get_corrected_outputs(vocs, train_y, train_c)
    feas = torch.all(corrected_train_c < 0.0, dim=-1).reshape(-1, 1)
    constraint_status = corrected_train_c < 0.0
    return feas, constraint_status


def check_training_data_shape(train_x, train_y, train_c, vocs):
    # check to make sure that the training tensor have the correct shapes
    for ele, vocs_type in zip([train_x, train_y, train_c], ['variables', 'objectives', 'constraints']):
        assert ele.ndim == 2, f'training data for vocs "{vocs_type}" must be 2 dim, shape currently is {ele.shape}'
        assert ele.shape[-1] == len(vocs[vocs_type]), f'current shape of training data ({ele.shape}) ' \
                                                      f'does not match number of vocs {vocs_type} == ' \
                                                      f'{len(vocs[vocs_type])} '

