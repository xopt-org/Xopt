import logging
import os
import sys

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
                      output_path=None,
                      custom_model=None,
                      executor=DummyExecutor(),
                      restart_file=None,
                      initial_x=None,
                      n_initial_samples=1,
                      n_steps=1,
                      verbose=True,
                      use_gpu=False,
                      ):
    """

    Parameters
    ----------
    use_gpu
    vocs
    verbose
    n_steps
    n_initial_samples
    initial_x
    executor
    restart_file
    custom_model
    output_path
    candidate_generator
    config : dict
        Varabiles, objectives, constraints and statics dictionary, see xopt documentation for detials

    evaluate_f : callable
        Returns dict of outputs after problem has been evaluated

    gen_candidate : callable
        Callable that takes the form f(model, bounds, vocs, **kwargs) and returns 1 or more candidates

    **kwargs
        Optional arguments for acquisition function optimization

    Returns
    -------
    results : dict
        Dictionary object containing optimization points + other info

    """

    # set up gpu if requested
    if use_gpu:
        if torch.cuda.is_available():
            tkwargs['device'] = torch.device('cuda')
        else:
            logger.warning('gpu requested but not found, using cpu')

    # Verbose print helper
    def vprint(*a, **k):
        # logger.debug(' '.join(a))
        # TODO: use logging instead of print statements
        if verbose:
            print(*a, **k)
            sys.stdout.flush()

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
        initial_y = [executor.submit(sampler_evaluate,
                                     dict(zip(variable_names, x.cpu().numpy())),
                                     evaluate_f,
                                     **sampler_evaluate_args) for x in initial_x]

        train_x, train_y, train_c = collect_results(initial_y, vocs, **tkwargs)

    else:
        train_x, train_y, train_c = get_data_json(restart_file,
                                                  vocs, **tkwargs)

    # do optimization
    for i in range(n_steps):

        # get corrected values
        corrected_train_y, corrected_train_c = get_corrected_outputs(vocs, train_y, train_c)

        # standardize y training data - use xopt version to allow for nans
        standardized_train_y = standardize(corrected_train_y)

        # horiz. stack objective and constraint results for training/acq specification
        train_outputs = torch.hstack((standardized_train_y, corrected_train_c))

        model = create_model(train_x, train_outputs, input_normalize, custom_model)

        # get candidate point(s)
        candidates = candidate_generator.generate(model, bounds, vocs, **tkwargs)

        if verbose:
            vprint(f'Candidate(s): {candidates}')

        # observe candidates
        fut = [executor.submit(sampler_evaluate,
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
            _, corrected_train_c = get_corrected_outputs(vocs, train_y, train_c)
            feas = torch.all(corrected_train_c < 0.0, dim=-1).reshape(-1, 1)
            constraint_status = corrected_train_c < 0.0

            full_data = torch.hstack((train_x, train_y, train_c, constraint_status, feas))
            save_data_dict(vocs, full_data, output_path)

        except NoValidResultsError:
            print('No valid results found, skipping to next iteration')
            continue

    # horiz. stack objective and constraint results for training/acq specification
    train_outputs = torch.hstack((train_y, train_c))

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
