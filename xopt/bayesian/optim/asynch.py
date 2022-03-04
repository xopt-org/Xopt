import concurrent.futures
import logging
import queue
from concurrent.futures import Executor
from typing import Dict, Optional, Callable

import torch
from botorch.utils.sampling import draw_sobol_samples

from ..data import gather_and_save_training_data, get_data_json
from ..generators.generator import BayesianGenerator
from ..utils import get_candidates, submit_candidates
from ...vocs_tools import get_bounds


def asynch(
    vocs: Dict,
    evaluate: Callable,
    processes: int,
    budget: float,
    candidate_generator: BayesianGenerator,
    executor: Executor,
    base_cost: float = 1.0,
    output_path: str = "",
    restart_file: Optional[str] = None,
    initial_x: Optional[torch.Tensor] = None,
    custom_model: Optional[Callable] = None,
    tkwargs: Optional[Dict] = None,
    logger: logging.Logger = None,
) -> tuple:

    # check if variable cost is specified in VOCS
    fixed_cost = False
    if "cost" not in vocs.variables:
        fixed_cost = True

    # define queue for evaluation
    q = queue.Queue()

    # track total cost of submitted candidates
    total_cost = 0

    # get data from previous runs, otherwise start with some initial samples
    if restart_file is None:
        # generate initial samples if no initial samples are given
        if initial_x is None:
            initial_x = draw_sobol_samples(
                torch.tensor(get_bounds(vocs), **tkwargs), 1, processes
            )[0]
        else:
            initial_x = initial_x

        # add initial points to the queue
        for ele in initial_x:
            q.put(ele)

        train_x, train_y, train_c, inputs, outputs = [None] * 5

    else:
        data = get_data_json(restart_file, vocs, **tkwargs)
        train_x = data["variables"]
        train_y = data["objectives"]
        train_c = data["constraints"]
        inputs = data["inputs"]
        outputs = data["outputs"]

        # get a new set of candidates and put them in the queue
        logger.info(f"generating {processes} new candidate(s) from restart file")
        new_candidates = get_candidates(
            train_x,
            train_y,
            vocs,
            candidate_generator,
            train_c=train_c,
            custom_model=custom_model,
            q=processes,
        )

        for ele in new_candidates:
            q.put(ele)

    # do optimization
    logger.info("starting optimization loop")

    futures = {}
    candidate_index = 0
    exceeded_budget = False

    while True:
        # check for status of the futures which are currently working
        done, not_done = concurrent.futures.wait(
            futures, timeout=5.0, return_when=concurrent.futures.FIRST_COMPLETED
        )

        # if there is incoming work, start a new future - unless our computation
        # budget has been exceeded
        while not q.empty() and not exceeded_budget:
            candidate = q.get()

            futures.update(
                submit_candidates(
                    candidate, executor, vocs, evaluate, {}, candidate_index
                )
            )

            if fixed_cost:
                c = 1.0
            else:
                c = candidate[-1] + base_cost
            total_cost += c

            logger.info(
                f"Submitted candidate {candidate_index:3}, cost: {c:4.3}, "
                f"total cost: {total_cost:4.4}"
            )
            logger.debug(f"{candidate}")

            candidate_index += 1

            if total_cost > budget:
                logger.info(f"budget exceeded, waiting for simulations to end")
                exceeded_budget = True
                break

        # process done futures and add new candidates to queue
        if len(done):
            # add results to dataset
            data = gather_and_save_training_data(
                list(done),
                vocs,
                tkwargs,
                train_x,
                train_y,
                train_c,
                inputs,
                outputs,
                output_path,
            )
            train_x, train_y, train_c, inputs, outputs = data

            # get a new set of candidates if budget has not been met and put them in
            # the queue
            if not exceeded_budget:
                logger.info(f"generating {len(done)} new candidate(s)")

                # get X_pending is available
                if len(not_done):
                    X_pending = []
                    for ele in list(not_done):
                        X_pending += [futures[ele].reshape(1, -1)]
                    candidate_generator.X_pending = torch.vstack(X_pending)

                new_candidates = get_candidates(
                    train_x,
                    train_y,
                    vocs,
                    candidate_generator,
                    train_c=train_c,
                    custom_model=custom_model,
                    q=len(done),
                )

                # add new candidates to queue
                logger.debug("Adding candidates to queue")
                for ele in new_candidates:
                    q.put(ele)

            # delete done futures from list
            for ele in done:
                del futures[ele]

        # end the optimization loop if we have run out of futures (they are all done)
        if not futures and exceeded_budget:
            logger.info("Budget exceeded and simulations finished")
            break

    return train_x, train_y, train_c, inputs, outputs
