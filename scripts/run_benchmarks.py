import argparse
import pathlib
import sys

import pandas as pd
import pytest

import torch

from xopt.resources.benchmarking import BenchMOBO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--direct", action="store_true", help="Run benchmarks without pytest"
    )
    parser.add_argument(
        "--steps", default=20, type=int, help="Number of optimization steps"
    )
    parser.add_argument(
        "--repeats", default=3, type=int, help="Number of reruns of each config"
    )
    parser.add_argument("--dump", default=None, type=str, help="Dump results to file")
    args = parser.parse_args()

    script_folder = pathlib.Path(__file__).parent

    if not args.direct:
        # This mode is for running from pytest, i.e. github actions, etc.
        # Pytest will automatically determine the number of times to run
        args = [
            "-vrxs",
            f"--benchmark-json={script_folder/'bench_output.json'}",
            f"{script_folder.parent/'benchmarks'}",
        ]

        # Add extra arguments
        if len(sys.argv) > 1:
            args.extend(sys.argv[1:])

        print(f"Running benchmarks on Python {sys.version}")
        print(f"pytest benchmark arguments: {args}")
        sys.exit(pytest.main(args))

    else:
        # This mode is for running/debugging directly, i.e. on a local machine
        torch.set_num_threads(1)
        df_bench_mobo = BenchMOBO.crate_parameter_table()
        bmobo = BenchMOBO(df_bench_mobo)
        bmobo.N_STEPS = args.steps
        N_REPEATS = 2
        results = []
        for row in df_bench_mobo.index.to_list():
            for r in range(N_REPEATS):
                outputs = bmobo.run(row)
                results.append(outputs)

        df = pd.DataFrame(results)
        dfgroup = df.groupby(["k"])

        dfsummary = dfgroup.mean(numeric_only=True)
        tidx = dfsummary.columns.get_loc("t")
        dfsummary.insert(loc=tidx + 1, column="t_std", value=dfgroup["t"].std())
        dfsummary["hvf_std"] = dfgroup["hvf"].std()
        dfsummary["hvf_min"] = dfgroup["hvf"].min()
        dfsummary["hvf_max"] = dfgroup["hvf"].max()

        # TODO: add tabulate for pretty-printing
        pd.set_option("display.max_columns", None)
        print(dfsummary.to_string(max_colwidth=20, float_format="{:.3f}".format))

        if args.dump is not None:
            dfsummary.to_csv(f"{script_folder/args.dump}")
