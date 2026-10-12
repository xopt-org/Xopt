from ..base import Xopt
from ..evaluator import DummyExecutor
from ..pydantic import remove_none_values
from .utils import (
    merge_dicts,
    normalize_initial_data,
    override_to_dict,
    setup_import_paths,
)
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
import argparse
import logging
import os
import pandas as pd
import yaml


logger = logging.getLogger(__name__)


@contextmanager
def get_executor(name, max_workers=1):
    """
    Context manager that returns the appropriate executor based on name.

    Parameters
    ----------
    name : str
        The executor type ('map', 'ThreadPoolExecutor', 'ProcessPoolExecutor')
    max_workers: int
        Number of workers/threads/processes

    Yields
    ------
    Executor
        The executor selected by name
    """
    if name is None:
        yield None
    elif name == "map":
        yield DummyExecutor()
    elif name == "thread_pool":
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            yield executor
    elif name == "process_pool":
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            yield executor
    else:
        raise ValueError(f"Unknown executor: {name}")


def main():
    # Handle the CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="The Xopt YAML config file")
    parser.add_argument(
        "--executor",
        help="Override the executor (and forcing vectorized=False)",
        type=str,
        choices=["map", "thread_pool", "process_pool"],
        default=None,
    )
    parser.add_argument(
        "--max_workers",
        help="Override number of workers (number of evaluations each time Xopt.step() is called)",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--python_path",
        help="Additional path to add to Python import path for evaluation function module search",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--override",
        help="Override config values using dot notation (e.g., generator.mutation_operator.eta_m=20)",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--initial_data",
        help="CSV file with initial data to seed the generator before running",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    args = parser.parse_args()

    # Start logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    setup_import_paths([os.getcwd()] + args.python_path)

    # Create xopt
    with open(args.config) as f:
        # Open file
        config = yaml.safe_load(f)

        # Clean up (replicate behavior of Xopt.from_file)
        config = remove_none_values(config)

        # Apply the overrides to the config dict
        if args.override:
            logger.info("Applying config file overrides:")
        for override in args.override:
            # Merge in the config override
            logger.info(f"  {override}")
            config = merge_dicts(config, override_to_dict(override))

        # Construct Xopt object
        my_xopt = Xopt.model_validate(config)

    # Get our executor and start xopt
    if args.executor is not None:
        msg = f"Starting Xopt with executor {args.executor}"
        if args.max_workers is not None:
            msg = msg + f" (max_workers={args.max_workers})"
        logger.info(msg)
    with get_executor(args.executor, max_workers=args.max_workers) as executor:
        # Handle executor override
        if args.executor is not None:
            my_xopt.evaluator.executor = executor
            my_xopt.evaluator.vectorized = False

        # Handle max_worker override
        if args.max_workers is not None:
            my_xopt.evaluator.max_workers = args.max_workers

        # Seed the generator with initial data if provided
        if args.initial_data is not None:
            logger.info(f"Loading initial data from {args.initial_data}")
            initial_df = normalize_initial_data(
                pd.read_csv(args.initial_data), my_xopt.vocs
            )
            my_xopt.add_data(initial_df)

        # Run Xopt
        my_xopt.run()


if __name__ == "__main__":
    main()
