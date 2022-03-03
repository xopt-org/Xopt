import logging
from concurrent.futures import Executor
from typing import Dict, Optional, Callable

import torch

from .generators.generator import BayesianGenerator
from .models.models import create_model
from .optim.asynch import asynch
from .optim.synch import synch
from .utils import get_corrected_outputs, get_feasability_constraint_status
from ..tools import DummyExecutor
from xopt.vocs import VOCS


"""
    Main optimization function for Bayesian optimization

"""

logger = logging.getLogger(__name__)


def optimize(
    vocs: Dict,
    evaluate_f: Callable,
    candidate_generator: BayesianGenerator,
    n_steps: int = None,
    n_initial_samples: int = None,
    processes: int = None,
    budget: float = None,
    base_cost: float = None,
    output_path: Optional[str] = "",
    custom_model: Optional[Callable] = None,
    executor: Optional[Executor] = None,
    restart_file: Optional[str] = None,
    initial_x: Optional[torch.Tensor] = None,
    tkwargs: Optional[Dict] = None,
) -> Dict:
    """
    Backend function for model based optimization

    Parameters
    ----------
    processes
    budget
    base_cost
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

    tkwargs : dict
        Specify data type and device for pytorch

    Returns
    -------
    results : dict
        Dictionary with output data at the end of optimization
    """
    
    
    vocs = VOCS.parse_obj(vocs) 
    # raise error if someone tries to use linked variables
    # TODO: implement linked variables
    if vocs.linked_variables:
        raise NotImplementedError(f"linked variables not implemented yet: {vocs.linked_variables}")

    # Handle None, False -> {}
    tkwargs = tkwargs or {}

    # check arguments for synch or asynch optimization
    use_synch = use_asynch = False
    if all([1 if ele is not None else 0 for ele in [n_steps, n_initial_samples]]):
        use_synch = True
    elif all([1 if ele is not None else 0 for ele in [budget, processes, base_cost]]):
        use_asynch = True
    else:
        raise RuntimeError(
            "must specify either n_steps and n_initial_samples or "
            "budget, processes, and base_cost"
        )

    # set executor
    executor = DummyExecutor() if executor is None else executor

    ##########################################
    # Do optimization
    ##########################################
    logger.info(f"started running optimization with generator: {candidate_generator}")
    if use_synch:
        result = synch(
            vocs,
            evaluate_f,
            n_initial_samples,
            n_steps,
            candidate_generator,
            executor,
            output_path,
            restart_file,
            initial_x,
            custom_model,
            tkwargs,
            logger,
        )
    elif use_asynch:
        result = asynch(
            vocs,
            evaluate_f,
            processes,
            budget,
            candidate_generator,
            executor,
            base_cost,
            output_path,
            restart_file,
            initial_x,
            custom_model,
            tkwargs,
            logger,
        )
    else:
        result = None

    train_x, train_y, train_c, inputs, outputs = result

    # horiz. stack objective and constraint results for training/acq specification
    feas, constraint_status = get_feasability_constraint_status(train_y, train_c, vocs)
    corrected_train_y, corrected_train_c = get_corrected_outputs(vocs, train_y, train_c)

    # output model
    model = create_model(
        train_x, corrected_train_y, corrected_train_c, vocs, custom_model
    )

    results = {
        "variables": train_x.cpu(),
        "objectives": train_y.cpu(),
        "corrected_objectives": corrected_train_y.cpu(),
        "constraint_status": constraint_status.cpu(),
        "feasibility": feas.cpu(),
        "model": model.cpu(),
    }

    if train_c is not None:
        results.update(
            {
                "constraints": train_c.cpu(),
                "corrected_constraints": corrected_train_c.cpu(),
            }
        )

    return results
