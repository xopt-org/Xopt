import concurrent.futures
import logging
import queue
from concurrent.futures import Executor
from typing import Dict, Optional, Callable

import torch
from botorch.utils.sampling import draw_sobol_samples

from .data import gather_and_save_training_data, get_data_json
from .generators.generator import BayesianGenerator
from .models.models import create_model
from .utils import get_corrected_outputs, get_candidates, \
    get_feasability_constraint_status, submit_candidates
from ..tools import DummyExecutor
from ..vocs_tools import get_bounds

"""
    Asynchronized optimization function for Bayesian optimization

"""

# Logger
logger = logging.getLogger(__name__)


def asynch_optimize(vocs: Dict,
                    evaluate_f: Callable,
                    candidate_generator: BayesianGenerator,
                    budget: float,
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
    Backend function for parallelized asynchronous optimization.

    As opposed to normal optimizers that follow the pattern "generate candidates" ->
    "evaluate candidates" -> "repeat", this optimizer generates candidates after each
    evaluation function returns a result. We assume that each call to the evaluate
    function will take a variable amount of time thus making normal optimization
    loops inefficient as the slowest evaulations dominate the algorithm runtime.

    To solve this problem we add candiates to a queue that is consumed by a
    parallelized executor (can also be done with serial evaluators). Candidates are
    generated and added to the queue every time we detect an evaluation has finished,
    meaning that roughly the same number of processes is used at all times.

    A stopping condition is specified by a maximum evaluation cost or `budget`. Each
    time an evaluation is submitted to the queue, we add the evaulation `cost` to the
    `total_cost`. If total_cost exceeds the budget, evaluations are no longer added
    to the queue and the optimizer waits for the remaining evaluations to finish.

    Notes
    ----------
    If `cost` is not specifed in `vocs['variables']` we assume a fixed cost of 1 for
    each evaluation, meaning `budget` reduces to the number of candidates that can be
    evaluated during optimization. In this case, `cost` is not supplied to the
    evaluate function.

    Parameters
    ----------
    vocs : dict
        Varabiles, objectives, constraints and statics dictionary,
        see xopt documentation for detials

    evaluate_f : callable
        Returns dict of outputs after problem has been evaluated

    candidate_generator : object
        Generator object that has a generate(model, bounds, vocs, **tkwargs) method

    budget : float
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

    # check if variable cost is specified in VOCS
    fixed_cost = False
    if 'cost' not in vocs['variables']:
        fixed_cost = True

    sampler_evaluate_args = {'verbose': verbose}

    # define queue for evaluation
    q = queue.Queue()

    # track total cost of submitted candidates
    total_cost = 0

    # get data from previous runs, otherwise start with some initial samples
    if restart_file is None:
        # generate initial samples if no initial samples are given
        if initial_x is None:
            initial_x = draw_sobol_samples(torch.tensor(get_bounds(vocs), **tkwargs),
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
        new_candidates = get_candidates(data[0],
                                        data[1],
                                        data[2],
                                        vocs,
                                        custom_model,
                                        candidate_generator,
                                        processes, )

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

            futures.update(submit_candidates(candidate,
                                             executor,
                                             vocs,
                                             evaluate_f,
                                             sampler_evaluate_args,
                                             candidate_index))

            if fixed_cost:
                c = 1.0
            else:
                c = candidate[-1] + base_cost
            total_cost += c

            logger.info(f'Submitted candidate {candidate_index}, cost: {c}')
            logger.debug(f'{candidate}')
            logger.info(f'total cost: {total_cost}')

            candidate_index += 1

            if total_cost > budget:
                logger.info(f'budget exceeded, waiting for simulations to end')
                exceeded_budget = True
                break

        # process done futures and add new candidates to queue
        if len(done):
            # add results to dataset
            data = gather_and_save_training_data(list(done),
                                                 vocs,
                                                 tkwargs,
                                                 data[0],
                                                 data[1],
                                                 data[2],
                                                 data[3],
                                                 data[4],
                                                 output_path)

            # get a new set of candidates if budget has not been met and put them in
            # the queue
            if not exceeded_budget:
                logger.info(f'generating {len(done)} new candidates')
                new_candidates = get_candidates(data[0],
                                                data[1],
                                                data[2],
                                                vocs,
                                                custom_model,
                                                candidate_generator,
                                                len(done), )

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
                                                                 data[2], )

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

