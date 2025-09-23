from .base import Xopt
from .evaluator import DummyExecutor
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
import argparse


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
    elif name == "ThreadPoolExecutor":
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            yield executor
    elif name == "ProcessPoolExecutor":
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            yield executor
    else:
        raise ValueError(f"Unknown executor: {name}")


def main():
    # Handle the CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", help="The Xopt YAML config file"
    )
    parser.add_argument(
        "--executor", 
        help="Override the executor (and forcing vectorized=False)", 
        type=str, 
        choices=["map", "ThreadPoolExecutor", "ProcessPoolExecutor"],
        default=None,
    )
    parser.add_argument(
        "--max_workers",
        help="Override number of workers (processes/threads/vectorized tasks)",
        type=int,
        default=None
    )
    args = parser.parse_args()

    # Create xopt
    with open(args.config) as f:
        my_xopt = Xopt.from_yaml(f.read())

    # Run it
    with get_executor(args.executor, max_workers=args.max_workers) as executor:
        if args.executor is not None:
            my_xopt.evaluator.executor = executor
            my_xopt.evaluator.vectorized = False
        if args.max_workers is not None:
            my_xopt.evaluator.max_workers = args.max_workers
        my_xopt.run()


if __name__ == "__main__":
    main()
