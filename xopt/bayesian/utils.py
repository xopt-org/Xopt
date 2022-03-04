import logging
import time
import traceback
from concurrent.futures import Executor
from typing import Dict, Optional, Tuple, Callable

import torch
from torch import Tensor

from .models.models import create_model


class NoValidResultsError(Exception):
    pass


class UnsupportedError(Exception):
    pass


# Logger
logger = logging.getLogger(__name__)

algorithm_defaults = {
    "n_steps": 30,
    "executor": None,
    "n_initial_samples": 5,
    "custom_model": None,
    "output_path": None,
    "verbose": True,
    "restart_data_file": None,
    "initial_x": None,
    "use_gpu": False,
    "eval_args": None,
}


def get_candidates(
    train_x: Tensor,
    train_y: Tensor,
    vocs: Dict,
    candidate_generator,
    custom_model: Optional[Callable] = None,
    train_c: Optional[Tensor] = None,
    q: Optional[int] = None,
):
    """
    Gets candidates based on training data

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
    corrected_train_y, corrected_train_c = get_corrected_outputs(vocs, train_y, train_c)
    # create and train model
    model_start = time.time()
    model = create_model(
        train_x, corrected_train_y, corrected_train_c, vocs, custom_model
    )
    logger.debug(f"Model creation time: {time.time() - model_start:.4} s")

    # get candidate point(s)
    candidate_start = time.time()
    candidates = candidate_generator.generate(model, q)
    logger.debug(f"Candidate generation time: {time.time() - candidate_start:.4} s")
    logger.debug(f"Candidate(s): {candidates}")
    return candidates


def submit_candidates(
    candidates: Tensor,
    executor: Executor,
    vocs: Dict,
    evaluate_f: Callable,
    sampler_evaluate_args: Dict,
    candidate_index_start: Optional[int] = 1,
) -> Dict:
    variable_names = list(vocs.variables)

    # add an extra axis if there is only one candidate
    candidates = torch.atleast_2d(candidates)

    fut = {}
    for candidate in candidates:
        setting = dict(zip(variable_names, candidate.cpu().numpy()))
        if vocs.constants is not None:
            setting.update(vocs.constants)

        fut.update(
            {
                executor.submit(
                    sampler_evaluate, setting, evaluate_f, **sampler_evaluate_args
                ): candidate
            }
        )

    return fut


def check_training_data_shape(
    train_x: [Tensor], train_y: [Tensor], train_c: [Tensor], vocs: [Dict]
) -> None:
    # check to make sure that the training tensor have the correct shapes
    for ele, vocs_type in zip(
        [train_x, train_y, train_c], ["variables", "objectives", "constraints"]
    ):
        if ele is not None:
            assert ele.ndim == 2, (
                f'training data for vocs "{vocs_type}" must be 2 dim, '
                f"shape currently is {ele.shape} "
            )

            assert ele.shape[-1] == len(getattr(vocs, vocs_type)), (
                f"current shape of training "
                f"data ({ele.shape}) "
                f"does not match number of vocs {vocs_type} == "
                f"{len(getattr(vocs, vocs_type))}"
            )
        else:
            if getattr(vocs, vocs_type):
                raise RuntimeError(
                    f"Training data for `{vocs_type}` is empty, but  `vocs['{vocs_type}'] = {vocs[vocs_type]}`"
                )


def get_feasability_constraint_status(
    train_y: [Tensor], train_c: [Tensor], vocs: [Dict]
) -> Tuple[Tensor, Tensor]:
    if train_c is not None:
        _, corrected_train_c = get_corrected_outputs(vocs, train_y, train_c)
        feas = torch.all(corrected_train_c < 0.0, dim=-1).reshape(-1, 1)
        constraint_status = corrected_train_c < 0.0
    else:
        feas = torch.ones((len(train_y), 1))
        constraint_status = feas.clone()
    return feas, constraint_status


def get_corrected_outputs(vocs, train_y, train_c):
    """
    scale and invert outputs depending on maximization/minimization, etc.
    """

    objectives = vocs.objectives
    objective_names = list(objectives.keys())

    # need to multiply -1 for each axis that we are using 'MINIMIZE' for an objective
    # need to multiply -1 for each axis that we are using 'GREATER_THAN' for a
    # constraint
    corrected_train_y = train_y.clone()

    # negate objective measurements that want to be minimized
    for j, name in zip(range(len(objective_names)), objective_names):
        if vocs.objectives[name] == "MINIMIZE":
            corrected_train_y[:, j] = -train_y[:, j]

        # elif vocs.objectives[name] == 'MAXIMIZE' or vocs.objectives[name] == 'None':
        #    pass
        else:
            pass
            # logger.warning(f'Objective goal {vocs.objectives[name]} not found, defaulting to MAXIMIZE')

    if train_c is not None:
        constraints = vocs.constraints
        constraint_names = list(constraints.keys())
        corrected_train_c = train_c.clone()

        # negate constraints that use 'GREATER_THAN'
        for k, name in zip(range(len(constraint_names)), constraint_names):
            if vocs.constraints[name][0] == "GREATER_THAN":
                corrected_train_c[:, k] = vocs.constraints[name][1] - train_c[:, k]

            elif vocs.constraints[name][0] == "LESS_THAN":
                corrected_train_c[:, k] = -(
                    vocs.constraints[name][1] - train_c[:, k]
                )
            else:
                logger.warning(
                    f'Constraint goal {vocs.constraints[name]} not found, defaulting to LESS_THAN'
                )
    else:
        corrected_train_c = None

    return corrected_train_y, corrected_train_c


def sampler_evaluate(inputs, evaluate_f, *eval_args, verbose=False):
    """
    Wrapper to catch any exceptions

    inputs: possible inputs to evaluate_f (a single positional argument)

    evaluate_f: a function that takes a dict with keys, and returns some output

    """
    outputs = None
    result = {}

    err = False
    try:
        outputs = evaluate_f(inputs, *eval_args)

    except Exception as ex:
        # No need to print a nasty exception
        logger.error(f"Exception caught in {__name__}")
        outputs = {
            "Exception": str(ex),
            "Traceback": traceback.print_tb(ex.__traceback__),
        }
        logger.error(outputs)
        err = True

    finally:
        result["inputs"] = inputs
        result["outputs"] = outputs
        result["error"] = err

    return result


def get_results(futures):
    # check the status of all futures
    results = []
    done = False
    ii = 1
    n_samples = len(futures)

    while True:
        logger.debug(f"futures length {len(futures)}")
        if len(futures) == 0:
            break
        else:
            # get the first element of futures - if done delete the element
            fut = futures[0]
            if fut.done():
                results.append(fut.result())
                del futures[0]

        # Slow down polling. Needed for MPI to work well.
        time.sleep(0.001)

    return results


def collect_results(futures, vocs, **tkwargs):
    """
    Collect successful measurement results into torch tensors to add to training data
    """

    train_x = []
    train_y = []

    if vocs.constraints is not None:
        train_c = []
    else:
        train_c = None

    inputs = []
    outputs = []

    at_least_one_point = False
    results = get_results(futures)

    for result in results:
        if not result["error"]:
            train_x += [[result["inputs"][ele] for ele in vocs.variables.keys()]]
            train_y += [[result["outputs"][ele] for ele in vocs.objectives.keys()]]

            if vocs.constraints is not None:
                train_c += [
                    [result["outputs"][ele] for ele in vocs.constraints.keys()]
                ]

            inputs += [result["inputs"]]
            outputs += [result["outputs"]]

            at_least_one_point = True

    if not at_least_one_point:
        raise NoValidResultsError("No valid results")

    train_x = torch.tensor(train_x, **tkwargs)
    train_y = torch.tensor(train_y, **tkwargs)
    if vocs.constraints is not None:
        train_c = torch.tensor(train_c, **tkwargs)

    logger.debug("done collecting results")
    return train_x, train_y, train_c, inputs, outputs
