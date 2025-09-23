from .base import Xopt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
import argparse


class MapExecutor:
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def map(self, func, iterable):
        return map(func, iterable)
    
    def submit(self, func, *args):
        from concurrent.futures import Future
        future = Future()
        try:
            result = func(*args)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        return future


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
    if name == "map":
        # Built-in map doesn't need a context manager, but we'll create a dummy one
        yield MapExecutor()
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
        help="The executor to use", 
        type=str, 
        choices=["map", "ThreadPoolExecutor", "ProcessPoolExecutor"],
        default="ThreadPoolExecutor",
    )
    parser.add_argument(
        "--n_cpu", help="Number of threads to launch.", type=int, default=1
    )
    args = parser.parse_args()

    # Create xopt
    with open(args.config) as f:
        my_xopt = X_from_yaml = Xopt.from_yaml(f.read())

    # Run it
    with get_executor(args.executor, max_workers=args.n_cpu) as executor:
        my_xopt.evaluator.executor = executor
        my_xopt.evaluator.max_workers = args.n_cpu
        my_xopt.run()


if __name__ == "__main__":
    main()
