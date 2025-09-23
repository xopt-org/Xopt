from .base import Xopt
from .evaluator import DummyExecutor
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
import argparse
import os
import sys


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
        default=None,
    )
    args = parser.parse_args()

    # Add specified path or CWD to sys.path
    import_path = args.python_path or os.getcwd()
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

    # Create xopt
    with open(args.config) as f:
        my_xopt = Xopt.from_yaml(f.read())

    # Get our executor
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
