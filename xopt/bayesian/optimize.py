import logging
import os
import sys
import time

import torch
from botorch.utils.sampling import draw_sobol_samples

from .data import save_data_dict, get_data_json
from .models.models import create_model
from .utils import get_bounds, collect_results, sampler_evaluate, \
    get_corrected_outputs, NoValidResultsError
from ..tools import full_path, DummyExecutor, isotime
from typing import Dict, Optional, Tuple, Callable, List
from .generators.generator import BayesianGenerator
from concurrent.futures import Executor, Future
from torch import Tensor

"""
    Main optimization function for Bayesian optimization

"""

# Logger
logger = logging.getLogger(__name__)


def optimize(vocs: [Dict],
             evaluate_f: [Callable],
             candidate_generator: [BayesianGenerator],
             n_steps: [int],
             n_initial_samples: [int],
             output_path: Optional[str] = '',
             custom_model: Optional[Callable] = None,
             executor: Optional[Executor] = None,
             restart_file: Optional[str] = None,
             initial_x: Optional[torch.Tensor] = None,
             verbose: Optional[bool] = False,
             tkwargs: Optional[Dict] = None,
             ) -> Dict:
    """
    Backend function for model based optimization

    Parameters
    ----------
    vocs : dict
        Varabiles, objectives, constraints and statics dictionary,
        see xopt documentation for detials

    evaluate_f : callable
        Returns dict of outputs after problem has been evaluated

    candidate_generator : object
        Generator object that has a generate(model, bounds, vocs, **tkwargs) method

    n_steps : int
        Number of optimization steps to execute

    n_initial_samples : int
        Number of initial samples to take before using the model,
        overwritten by initial_x

    output_path : str
        Path location to place outputs

    custom_model : callable
        Function of the form f(train_inputs, train_outputs) that
        returns a trained custom model

    executor : Executor
        Executor object to run evaluate_f

    restart_file : str
        File location of JSON file that has previous data

    initial_x : list
        Nested list to provide initial candiates to evaluate,
        overwrites n_initial_samples

    verbose : bool
        Print out messages during optimization

    tkwargs : dict
        Specify data type and device for pytorch

    Returns
    -------
    results : dict
        Dictionary with output data at the end of optimization
    """

    # raise error if someone tries to use linked variables
    # TODO: implement linked variables
    if 'linked_variables' in vocs.keys():
        assert vocs['linked_variables'] == {}, 'linked variables not implemented yet'

    if tkwargs is None:
        tkwargs = {}

    # Verbose print helper
    def vprint(*a, **k):
        # logger.debug(' '.join(a))
        # TODO: use logging instead of print statements
        if verbose:
            print(*a, **k)
            sys.stdout.flush()

    vprint(f'started running optimization with generator: {candidate_generator}')

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
    variable_names = list(variables)

    # create normalization transforms for model inputs
    # inputs are normalized in [0,1]

    sampler_evaluate_args = {'verbose': verbose}

    # generate initial samples if no initial samples are given
    if restart_file is None:
        if initial_x is None:
            initial_x = draw_sobol_samples(get_bounds(vocs, **tkwargs),
                                           1, n_initial_samples)[0]
        else:
            initial_x = initial_x

        # submit evaluation of initial samples
        vprint(f'submitting initial candidates at time {isotime()}')
        initial_y = submit_jobs(initial_x, exe, vocs, evaluate_f, sampler_evaluate_args)

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
        corrected_train_y, corrected_train_c = get_corrected_outputs(vocs,
                                                                     train_y,
                                                                     train_c)

        # create and train model
        model_start = time.time()
        model = create_model(train_x,
                             corrected_train_y,
                             corrected_train_c,
                             vocs,
                             custom_model)
        vprint(f'Model creation time: {time.time() - model_start:.4} s')

        # get candidate point(s)
        candidate_start = time.time()
        candidates = candidate_generator.generate(model)
        vprint(f'Candidate generation time: {time.time() - candidate_start:.4} s')
        vprint(f'Candidate(s): {candidates}')

        # observe candidates
        vprint(f'submitting candidates at time {isotime()}')
        fut = submit_jobs(candidates, exe, vocs, evaluate_f, sampler_evaluate_args)

        try:
            new_x, new_y, new_c = collect_results(fut, vocs, **tkwargs)

            # add new observations to training data
            train_x = torch.vstack((train_x, new_x))
            train_y = torch.vstack((train_y, new_y))
            train_c = torch.vstack((train_c, new_c))

            # get feasibility values
            feas, constraint_status = get_feasability_constraint_status(train_y,
                                                                        train_c,
                                                                        vocs)

            full_data = torch.hstack((train_x,
                                      train_y,
                                      train_c,
                                      constraint_status,
                                      feas))
            save_data_dict(vocs, full_data, output_path)

        except NoValidResultsError:
            print('No valid results found, skipping to next iteration')
            continue

    # horiz. stack objective and constraint results for training/acq specification
    feas, constraint_status = get_feasability_constraint_status(train_y, train_c, vocs)
    corrected_train_y, corrected_train_c = get_corrected_outputs(vocs, train_y, train_c)

    # output model
    model = create_model(train_x, corrected_train_y, corrected_train_c,
                         vocs, custom_model)

    results = {'variables': train_x.cpu(),
               'objectives': train_y.cpu(),
               'corrected_objectives': corrected_train_y.cpu(),
               'constraints': train_c.cpu(),
               'corrected_constraints': corrected_train_c.cpu(),
               'constraint_status': constraint_status.cpu(),
               'feasibility': feas.cpu(),
               'model': model.cpu()}

    return results


def get_feasability_constraint_status(train_y: [Tensor],
                                      train_c: [Tensor],
                                      vocs: [Dict]) -> Tuple[Tensor, Tensor]:
    _, corrected_train_c = get_corrected_outputs(vocs, train_y, train_c)
    feas = torch.all(corrected_train_c < 0.0, dim=-1).reshape(-1, 1)
    constraint_status = corrected_train_c < 0.0
    return feas, constraint_status


def submit_jobs(candidates: [Tensor],
                exe: [Executor],
                vocs: [Dict],
                evaluate_f: [Callable],
                sampler_evaluate_args: [Dict]) -> List[Future]:
    variable_names = list(vocs['variables'])
    settings = get_settings(candidates, variable_names, vocs)
    fut = [exe.submit(sampler_evaluate,
                      setting,
                      evaluate_f,
                      **sampler_evaluate_args) for setting in settings]
    return fut


def get_settings(X: [Tensor],
                 variable_names: List[str],
                 vocs: [Dict]) -> List[Dict]:
    settings = [dict(zip(variable_names, x.cpu().numpy())) for x in X]
    for setting in settings:
        setting.update(vocs['constants'])

    return settings


def check_training_data_shape(train_x: [Tensor],
                              train_y: [Tensor],
                              train_c: [Tensor],
                              vocs: [Dict]) -> None:
    # check to make sure that the training tensor have the correct shapes
    for ele, vocs_type in zip([train_x, train_y, train_c],
                              ['variables', 'objectives', 'constraints']):
        assert ele.ndim == 2, f'training data for vocs "{vocs_type}" must be 2 dim, ' \
                              f'shape currently is {ele.shape} '

        assert ele.shape[-1] == len(vocs[vocs_type]),\
                                    f'current shape of training '\
                                    f'data ({ele.shape}) '\
                                    f'does not match number of vocs {vocs_type} == '\
                                    f'{len(vocs[vocs_type])} '
