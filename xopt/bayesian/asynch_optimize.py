import concurrent.futures
import logging
import queue
import time
from concurrent.futures import Executor
from typing import Dict, Optional, Tuple, Callable, List

import torch
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor

from .data import save_data_dict, get_data_json
from .generators.generator import BayesianGenerator
from .models.models import create_model
from .utils import get_bounds, collect_results, sampler_evaluate, \
    get_corrected_outputs, NoValidResultsError
from ..tools import DummyExecutor

"""
    Asynchronized optimization function for Bayesian optimization

"""

# Logger
logger = logging.getLogger(__name__)


def asynch_optimize(vocs: Dict,
                    evaluate_f: Callable,
                    candidate_generator: BayesianGenerator,
                    budget: int,
                    processes: int,
                    base_cost: Optional[float] = 1.0,
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

    budget : int
        Optimization budget

    processes : int
        Number of processes to evaluate simultaneously

    base_cost : float, default=0.1
        Base cost for simulations

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
    logger.info(f'started running optimization with generator: {candidate_generator}')

    # set executor
    executor = DummyExecutor() if executor is None else executor

    # parse VOCS
    variables = vocs['variables']
    variable_names = list(variables)

    assert variable_names[-1] == 'cost'

    sampler_evaluate_args = {'verbose': verbose}

    # define queue for evaluation
    q = queue.Queue()

    # track total cost of submitted candidates
    total_cost = 0

    # get data from previous runs, otherwise start with some initial samples
    if restart_file is None:
        # generate initial samples if no initial samples are given
        if initial_x is None:
            initial_x = draw_sobol_samples(get_bounds(vocs, **tkwargs),
                                           1, processes)[0]
        else:
            initial_x = initial_x

        # add initial points to the queue
        for ele in initial_x:
            q.put(ele)

        data = [None] * 5

    else:
        data = get_data_json(restart_file,
                             vocs, **tkwargs)

        # get a new set of candidates and put them in the queue
        new_candidates = get_candidates(processes,
                                        *data[:3],
                                        vocs,
                                        custom_model,
                                        candidate_generator)

        for ele in new_candidates:
            q.put(ele)

    # do optimization
    logger.info('starting optimization loop')

    futures = {}
    candidate_index = 0
    exceeded_budget = False

    while True:
        # check for status of the futures which are currently working
        done, not_done = concurrent.futures.wait(
            futures, timeout=5.0,
            return_when=concurrent.futures.ALL_COMPLETED)

        # if there is incoming work, start a new future - unless our computation
        # budget has been exceeded
        while not q.empty() and not exceeded_budget:
            candidate = q.get()

            logger.info(f'Submitting candidate {candidate_index}: {candidate}')
            futures.update(submit_candidate(candidate,
                                            executor,
                                            vocs,
                                            evaluate_f,
                                            sampler_evaluate_args,
                                            candidate_index))
            candidate_index += 1
            total_cost += candidate[-1] + base_cost
            logger.info(f'total cost: {total_cost}')

            if total_cost > budget:
                logger.info(f'budget exceeded, waiting for simulations to end')
                exceeded_budget = True
                break

        # process done futures and add new candidates to queue
        if len(done):
            # add results to dataset
            data = gather(list(done),
                          vocs,
                          *data,
                          tkwargs,
                          output_path)

            # get a new set of candidates if budget has not been met and put them in
            # the queue
            if not exceeded_budget:
                logger.info(f'generating {len(done)} new candidates')
                new_candidates = get_candidates(len(done),
                                                *data[:3],
                                                vocs,
                                                custom_model,
                                                candidate_generator)

                for ele in new_candidates:
                    q.put(ele)

            # delete done futures from list
            for ele in done:
                del futures[ele]

        # end the optimization loop if we have run out of futures (they are all done)
        if not futures and exceeded_budget:
            logger.info('Budget exceeded and simulations finished')
            break

    # horiz. stack objective and constraint results for training/acq specification
    feas, constraint_status = get_feasability_constraint_status(data[1],
                                                                data[2],
                                                                vocs)
    corrected_train_y, corrected_train_c = get_corrected_outputs(vocs,
                                                                 data[1],
                                                                 data[2],)

    # output model
    model = create_model(data[0], corrected_train_y, corrected_train_c,
                         vocs, custom_model)

    results = {'variables': data[0].cpu(),
               'objectives': data[1].cpu(),
               'corrected_objectives': corrected_train_y.cpu(),
               'constraints': data[2].cpu(),
               'corrected_constraints': corrected_train_c.cpu(),
               'constraint_status': constraint_status.cpu(),
               'feasibility': feas.cpu(),
               'model': model.cpu()}

    return results


def get_candidates(q,
                   train_x,
                   train_y,
                   train_c,
                   vocs,
                   custom_model,
                   candidate_generator):
    """


    Parameters
    ----------
    custom_model
    candidate_generator
    q
    train_x
    train_y
    train_c
    vocs

    Returns
    -------

    """
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
    logger.info(f'Model creation time: {time.time() - model_start:.4} s')

    # get candidate point(s)
    candidate_start = time.time()
    candidates = candidate_generator.generate(model, q)
    logger.info(f'Candidate generation time: {time.time() - candidate_start:.4} s')
    logger.info(f'Candidate(s): {candidates}')
    return candidates


def gather(done,
           vocs,
           train_x,
           train_y,
           train_c,
           inputs,
           outputs,
           tkwargs,
           output_path):
    try:
        new_x, new_y, new_c, new_inputs, new_outputs = collect_results(done,
                                                                       vocs,
                                                                       **tkwargs)

        if train_x is None:
            train_x = new_x
            train_y = new_y
            train_c = new_c
            inputs = new_inputs
            outputs = new_outputs

        else:
            # add new observations to training data
            train_x = torch.vstack((train_x, new_x))
            train_y = torch.vstack((train_y, new_y))
            train_c = torch.vstack((train_c, new_c))

            inputs += new_inputs
            outputs += new_outputs

        # get feasibility values
        feas, constraint_status = get_feasability_constraint_status(train_y,
                                                                    train_c,
                                                                    vocs)

        full_data = torch.hstack((train_x,
                                  train_y,
                                  train_c,
                                  constraint_status,
                                  feas))
        save_data_dict(vocs, full_data, inputs, outputs, output_path)

    except NoValidResultsError:
        logger.warning('No valid results found, skipping to next iteration')

    return train_x, train_y, train_c, inputs, outputs


def get_feasability_constraint_status(train_y: [Tensor],
                                      train_c: [Tensor],
                                      vocs: [Dict]) -> Tuple[Tensor, Tensor]:
    _, corrected_train_c = get_corrected_outputs(vocs, train_y, train_c)
    feas = torch.all(corrected_train_c < 0.0, dim=-1).reshape(-1, 1)
    constraint_status = corrected_train_c < 0.0
    return feas, constraint_status


def submit_candidate(candidate: [Tensor],
                     executor: [Executor],
                     vocs: [Dict],
                     evaluate_f: [Callable],
                     sampler_evaluate_args: [Dict],
                     candidate_index_start: [int]) -> Dict:
    variable_names = list(vocs['variables'])
    settings = get_settings(candidate, variable_names, vocs)
    fut = {executor.submit(sampler_evaluate,
                           settings,
                           evaluate_f,
                           **sampler_evaluate_args): candidate_index_start + 1}

    return fut


def get_settings(X: [Tensor],
                 variable_names: List[str],
                 vocs: [Dict]) -> Dict:
    settings = dict(zip(variable_names, X.cpu().numpy()))
    settings.update(vocs['constants'])

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

        assert ele.shape[-1] == len(vocs[vocs_type]), \
            f'current shape of training ' \
            f'data ({ele.shape}) ' \
            f'does not match number of vocs {vocs_type} == ' \
            f'{len(vocs[vocs_type])}'
