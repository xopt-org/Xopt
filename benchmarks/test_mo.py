import logging

import pytest
import torch

from xopt.resources.benchmarking import BenchMOBO

df_bench_mobo = BenchMOBO.crate_parameter_table()
bmobo = BenchMOBO(df_bench_mobo)
bmobo.N_STEPS = 5


@pytest.mark.parametrize("row", df_bench_mobo.index.to_list())
@pytest.mark.benchmark(
    group="mobo",
    min_time=1.0,
    max_time=60.0,
    min_rounds=2,
    disable_gc=True,
    warmup=False,
)
def test_mobo(benchmark, row):
    torch.set_num_threads(1)
    print(f"Running benchmark for {df_bench_mobo.loc[row, :]}")
    logging.basicConfig(level=logging.DEBUG)
    result = benchmark(bmobo.run, row)
    benchmark.extra_info.update(result)
