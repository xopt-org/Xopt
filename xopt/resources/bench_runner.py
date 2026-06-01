import argparse

from xopt.resources.bench_framework import BenchDispatcher, BenchSuite
import xopt.resources.bench_functions  # noqa: F401


def run_benchmark():
    parser = argparse.ArgumentParser(description="Run benchmark function")
    parser.add_argument("-n", type=int, help="number of repeats", nargs="?", default=1)
    parser.add_argument("-device", type=str, help="device to use", default="cpu")
    parser.add_argument(
        "benchmarks", type=str, nargs="+", help="Name of the benchmark function to run"
    )
    args = parser.parse_args()

    dev_kwargs = {}
    device = args.device
    assert device == "cpu" or device == "cuda" or device.startswith("cuda:"), (
        "Device must be 'cpu', 'cuda', or 'cuda:<index>'"
    )
    if device == "cpu":
        dev_kwargs["device"] = "cpu"
    else:
        import torch

        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this machine")
        if device == "cuda":
            device = "cuda:0"
        dev_kwargs["device"] = device

    runner = BenchSuite()
    for benchmark in args.benchmarks:
        try:
            pre, func = BenchDispatcher.get(benchmark)
            kwargs = BenchDispatcher.get_kwargs(benchmark)
            kwargs.update(dev_kwargs)
            print(f"Adding benchmark {benchmark} for {args.n} iterations")
        except KeyError:
            raise ValueError(f"Benchmark function {benchmark} not found")
        runner.add(func, kwargs=kwargs, preamble=pre)
    runner.run(min_rounds=args.n, min_time=0.0)


if __name__ == "__main__":
    run_benchmark()
