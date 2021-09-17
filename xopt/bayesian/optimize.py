import logging
import os
import sys
from concurrent.futures import Executor
from typing import Dict, Optional, Callable

import torch
from botorch.utils.sampling import draw_sobol_samples

from .data import get_data_json, gather_and_save_training_data
from .generators.generator import BayesianGenerator
from .models.models import create_model
from .utils import submit_candidates, \
    get_corrected_outputs, get_candidates, \
    get_feasability_constraint_status
from ..tools import full_path, DummyExecutor, isotime
from ..vocs_tools import get_bounds

"""
    Main optimization function for Bayesian optimization

"""

# Logger
logger = logging.getLogger(__name__)


def optimize(vocs: Dict,
             evaluate_f: Callable,
             candidate_generator: BayesianGenerator,
             n_steps: int,
             n_initial_samples: int,
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
    executor = DummyExecutor() if executor is None else executor

    sampler_evaluate_args = {'verbose': verbose}

    # generate initial samples if no initial samples are given
    if restart_file is None:
        if initial_x is None:
            initial_x = draw_sobol_samples(torch.tensor(get_bounds(vocs),
                                                        **tkwargs),
                                           1, n_initial_samples)[0]
        else:
            initial_x = initial_x

        # submit evaluation of initial samples
        vprint(f'submitting initial candidates at time {isotime()}')
        initial_y = submit_candidates(initial_x, executor, vocs, evaluate_f,
                                      sampler_evaluate_args)

        data = gather_and_save_training_data(list(initial_y),
                                             vocs,
                                             tkwargs,
                                             output_path=output_path
                                             )
        train_x, train_y, train_c, inputs, outputs = data

    else:
        data = get_data_json(restart_file, vocs, **tkwargs)

        train_x = data['variables']
        train_y = data['objectives']
        train_c = data['constraints']
        inputs = data['inputs']
        outputs = data['outputs']

    # do optimization
    vprint('starting optimization loop')
    for i in range(n_steps):
        candidates = get_candidates(train_x,
                                    train_y,
                                    vocs,
                                    candidate_generator,
                                    train_c=train_c,
                                    custom_model=custom_model,
                                    )

        # observe candidates
        vprint(f'submitting candidates at time {isotime()}')
        fut = submit_candidates(candidates,
                                executor,
                                vocs,
                                evaluate_f,
                                sampler_evaluate_args)

        data = gather_and_save_training_data(list(fut),
                                             vocs,
                                             tkwargs,
                                             train_x,
                                             train_y,
                                             train_c,
                                             inputs,
                                             outputs,
                                             output_path=output_path
                                             )
        train_x, train_y, train_c, inputs, outputs = data

    # horiz. stack objective and constraint results for training/acq specification
    feas, constraint_status = get_feasability_constraint_status(train_y, train_c, vocs)
    corrected_train_y, corrected_train_c = get_corrected_outputs(vocs, train_y, train_c)

    # output model
    model = create_model(train_x, corrected_train_y, corrected_train_c,
                         vocs, custom_model)

    results = {'variables': train_x.cpu(),
               'objectives': train_y.cpu(),
               'corrected_objectives': corrected_train_y.cpu(),
               'constraint_status': constraint_status.cpu(),
               'feasibility': feas.cpu(),
               'model': model.cpu()}

    if train_c is not None:
        results.update({'constraints': train_c.cpu(),
                        'corrected_constraints': corrected_train_c.cpu(), })

    return results
