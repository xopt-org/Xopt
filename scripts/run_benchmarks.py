import pandas as pd

import torch

from xopt.resources.benchmarking import BenchMOBO

if __name__ == "__main__":
    torch.set_num_threads(1)
    df_bench_mobo = BenchMOBO.crate_parameter_table()
    bmobo = BenchMOBO(df_bench_mobo)
    bmobo.N_STEPS = 2
    N_REPEATS = 1
    results = []
    for row in df_bench_mobo.index.to_list():
        for r in range(N_REPEATS):
            outputs = bmobo.run(row)
            results.append(outputs)

    #df = pd.concat([df_bench_mobo, pd.json_normalize(df_bench_mobo['opts'])], axis=1)
    df = pd.DataFrame(results)
    #df.drop(columns=['rp'], inplace=True)
    dfgroup = df.groupby(['k'])

    dfsummary = dfgroup.mean(numeric_only=True)
    dfsummary['t_std'] = dfgroup['t'].std()
    dfsummary['hvf_std'] = dfgroup['hvf'].std()
    dfsummary['hvf_min'] = dfgroup['hvf'].min()
    dfsummary['hvf_max'] = dfgroup['hvf'].max()

    pd.set_option('display.max_columns', None)
    print(dfsummary)