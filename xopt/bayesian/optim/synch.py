import logging
from concurrent.futures import Executor
from typing import Dict, Optional, Callable

import torch
from botorch.utils.sampling import draw_sobol_samples

from ..data import gather_and_save_training_data, get_data_json
from ..generators.generator import BayesianGenerator
from ..utils import get_candidates, submit_candidates
from ...vocs_tools import get_bounds


def synch(
    vocs: Dict,
    evaluate: Callable,
    n_initial_samples: int,
    n_steps: int,
    candidate_generator: BayesianGenerator,
    executor: Executor,
    output_path: str = "",
    restart_file: Optional[str] = None,
    initial_x: Optional[torch.Tensor] = None,
    custom_model: Optional[Callable] = None,
    tkwargs: Optional[Dict] = None,
    logger: logging.Logger = None,
) -> tuple:

    # generate initial samples if no initial samples are given
    if restart_file is None:
        if initial_x is None:
            initial_x = draw_sobol_samples(
                torch.tensor(get_bounds(vocs), **tkwargs), 1, n_initial_samples
            )[0]
        else:
            initial_x = torch.tensor(initial_x)

        # submit evaluation of initial samples
        logger.info(f"submitting initial candidates")
        initial_y = submit_candidates(initial_x, executor, vocs, evaluate, {})

        data = gather_and_save_training_data(
            list(initial_y), vocs, tkwargs, output_path=output_path
        )
        train_x, train_y, train_c, inputs, outputs = data

    else:
        data = get_data_json(restart_file, vocs, **tkwargs)

        train_x = data["variables"]
        train_y = data["objectives"]
        train_c = data["constraints"]
        inputs = data["inputs"]
        outputs = data["outputs"]

    # do optimization
    logger.info("starting optimization loop")
    for i in range(n_steps):
        candidates = get_candidates(
            train_x,
            train_y,
            vocs,
            candidate_generator,
            train_c=train_c,
            custom_model=custom_model,
        )

        # observe candidates
        logger.info(f"submitting candidates")
        fut = submit_candidates(candidates, executor, vocs, evaluate, {})

        data = gather_and_save_training_data(
            list(fut),
            vocs,
            tkwargs,
            train_x,
            train_y,
            train_c,
            inputs,
            outputs,
            output_path=output_path,
        )
        train_x, train_y, train_c, inputs, outputs = data

    return train_x, train_y, train_c, inputs, outputs
