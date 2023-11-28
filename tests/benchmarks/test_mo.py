import logging

import pytest
import torch

from xopt.resources.benchmarking import BenchMOBO

df_bench_mobo = BenchMOBO.crate_parameter_table()
bmobo = BenchMOBO(df_bench_mobo)
bmobo.N_STEPS = 5

@pytest.mark.parametrize('row', df_bench_mobo.index.to_list())
@pytest.mark.benchmark(
        group="mobo",
        min_time=1.0,
        max_time=20.0,
        min_rounds=2,
        disable_gc=True,
        warmup=False,
)
def test_mobo(benchmark, row):
    torch.set_num_threads(1)

    logging.basicConfig(level=logging.DEBUG)
    result = benchmark(bmobo.run, row)
    benchmark.extra_info.update(result)

    # Group repeats
    # df = pd.concat([df, pd.json_normalize(df['opts'])], axis=1)
    # df.drop(columns=['function', 'opts', 'rp'], inplace=True)
    # dfgroup = df.groupby(['k'])
    #
    # dfsummary = dfgroup.mean(numeric_only=True)
    # dfsummary['t_std'] = dfgroup['t'].std()
    # dfsummary['hvf_std'] = dfgroup['hvf'].std()
    # dfsummary['hvf_min'] = dfgroup['hvf'].min()
    # dfsummary['hvf_max'] = dfgroup['hvf'].max()
    #
    # print(dfsummary)
