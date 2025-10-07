from .base import Xopt
from .evaluator import DummyExecutor
from .pydantic import remove_none_values
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
import argparse
import logging
import os
import sys
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


def override_to_dict(override: str) -> dict:
    """
    Convert strings of form "class_a.class_b.class_c.param=1" to
    {'class_a': {'class_b': {'class_c': {'param': 1}}}}.

    Uses yaml library for consistent type conversion of values following
    same convention as in config files.

    Parameters
    ----------
    override : str
        The override string

    Returns
    -------
    dict
        The nested dictionary containing the value.
    """
    path, value = override.split("=", 1)
    yaml_str = (
        "\n".join([" " * idx + x + ":" for idx, x in enumerate(path.split("."))])
        + " "
        + value.strip()
    )
    return yaml.safe_load(yaml_str)


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """
    Nested merging of dicts. Will combine nested dicts keeping keys in both with
    the values in dict2 overriding those in dict1.

    Parameters
    ----------
    dict1 : dict
        Parameters to override
    dict2 : dict
        Parameters used to override those in dict1

    Returns
    -------
    dict
        Values from dict1 with overrides from dict2
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


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
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    args = parser.parse_args()

    # Start logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Add specified paths and CWD to sys.path
    import_paths = [os.getcwd()]
    import_paths.extend(
        [os.path.expanduser(os.path.expandvars(x)) for x in args.python_path]
    )
    if len(import_paths) > 1:
        logger.info("Python path additions:")
    for idx, import_path in enumerate(import_paths):
        if idx:
            logger.info(f"  {import_path}")
        if import_path not in sys.path:
            sys.path.insert(0, import_path)

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
            logger.info(f"  {override}")
            # Sanity check
            if "=" not in override:
                raise ValueError(
                    f'Invalid override format: "{override}". Expected key=value'
                )

            # Merge in the config override
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

        # Run Xopt
        my_xopt.run()


if __name__ == "__main__":
    main()
