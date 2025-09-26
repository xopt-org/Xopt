import argparse

from xopt.resources.benchmarking import BenchDispatcher


def run_benchmark():
    parser = argparse.ArgumentParser(description="Run benchmark function")
    parser.add_argument("benchmark", type=str, help="Name of the benchmark function to run")
    parser.add_argument("-n", type=int, help="number of repeats", nargs='?', default=1)
    parser.add_argument('-device', type=str, help="device to use", default='cpu')
    args = parser.parse_args()

    try:
        pre, func = BenchDispatcher.get(args.benchmark)
        kwargs = BenchDispatcher.get_kwargs(args.benchmark)
        print(f"Running benchmark {args.benchmark} for {args.n} iterations")
    except KeyError:
        raise ValueError(f"Benchmark function {args.benchmark} not found")

    device = args.device
    assert device == 'cpu' or device == 'cuda' or device.startswith('cuda:'), "Device must be 'cpu', 'cuda', or 'cuda:<index>'"

    pre()
    for _ in range(args.n):
        func(**kwargs, device=args.device)


if __name__ == "__main__":
    run_benchmark()