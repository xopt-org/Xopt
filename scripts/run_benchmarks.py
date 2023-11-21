import time

import pandas as pd

import torch

from tqdm import tqdm
from xopt import Evaluator, Xopt
from xopt.generators.bayesian import MOBOGenerator
from xopt.resources.test_functions.dtlz import dtlz2_reference_point, dtlz2_vocs, evaluate_DTLZ2
from xopt.resources.test_functions.tnk import evaluate_TNK, tnk_reference_point, tnk_vocs

torch.set_num_threads(1)


# Can move into tests, but annoying to run
# benchmark = pytest.mark.skipif('--run-bench' not in sys.argv, reason="No benchmarking requested")
class BenchMOBO:
    KEYS = ['dtlz2', 'tnk']
    FUNCTIONS = {'dtlz2': evaluate_DTLZ2, 'tnk': evaluate_TNK}
    VOCS = {'dtlz2': dtlz2_vocs, 'tnk': tnk_vocs}
    RPS = {'dtlz2': dtlz2_reference_point, 'tnk': tnk_reference_point}
    REPEATS = {'dtlz2': 10, 'tnk': 5}
    N_STEPS = 4

    OPTS = [dict(n_monte_carlo_samples=16, log_transform_acquisition_function=False),
            dict(n_monte_carlo_samples=128, log_transform_acquisition_function=False),
            dict(n_monte_carlo_samples=16, log_transform_acquisition_function=True),
            dict(n_monte_carlo_samples=128, log_transform_acquisition_function=True),
            ]

    def run_opt(self, gen, evaluator, n_evals):
        """ Run xopt with specified generator and evaluator """
        X = Xopt(generator=gen, evaluator=evaluator, vocs=gen.vocs)
        X.random_evaluate(2)
        X.data.loc[:, 'gen_time'] = 0.0
        X.data.loc[:, 'hv'] = 0.0
        for i in tqdm(range(n_evals), position=1):
            # TODO: add internal timing to Xopt generators directly
            t1 = time.perf_counter()
            X.step()
            t2 = time.perf_counter()
            X.data.iloc[-1, X.data.columns.get_loc('gen_time')] = t2 - t1
            X.data.iloc[-1, X.data.columns.get_loc('hv')] = X.generator.calculate_hypervolume()
        return X

    def crate_parameter_table(self):
        """ Create a table of generator parameters to benchmark """
        rows = []
        for k in self.KEYS:
            for rep in range(self.REPEATS[k]):
                for i, opts in enumerate(self.OPTS):
                    rows.append({'k': f'{k}_{i}', 'fname': k, 'opts': opts, 'rp': self.RPS[k],
                                 'rep': rep,
                                 'function': self.FUNCTIONS[k], 'vocs': self.VOCS[k]
                                 })

        return pd.DataFrame(rows)

    def run(self):
        pd.set_option('display.max_columns', None)

        df = self.crate_parameter_table()
        print(f'Running {len(df)} benchmarks')
        print(df)

        results = {}
        for row in tqdm(df.index):
            evaluator = Evaluator(function=self.FUNCTIONS[df.loc[row, 'fname']])
            gen = MOBOGenerator(vocs=df.loc[row, 'vocs'], reference_point=df.loc[row, 'rp'],
                                **df.loc[row, 'opts'])
            X = self.run_opt(gen, evaluator, self.N_STEPS)
            results[row] = X

        df['t'] = [X.data.gen_time.sum() for X in results.values()]
        df['hv25'] = [X.data.loc[1 * self.N_STEPS // 4, 'hv'] for X in results.values()]
        df['hv50'] = [X.data.loc[2 * self.N_STEPS // 4, 'hv'] for X in results.values()]
        df['hv75'] = [X.data.loc[3 * self.N_STEPS // 4, 'hv'] for X in results.values()]
        df['hvf'] = [X.generator.calculate_hypervolume() for X in results.values()]

        print(f'\n================= DONE ===================\n')

        # Group repeats
        df = pd.concat([df, pd.json_normalize(df['opts'])], axis=1)
        df.drop(columns=['function', 'opts', 'rp'], inplace=True)
        dfgroup = df.groupby(['k'])

        dfsummary = dfgroup.mean(numeric_only=True)
        dfsummary['t_std'] = dfgroup['t'].std()
        dfsummary['hvf_std'] = dfgroup['hvf'].std()
        dfsummary['hvf_min'] = dfgroup['hvf'].min()
        dfsummary['hvf_max'] = dfgroup['hvf'].max()

        print(dfsummary)


class BenchSOBO:
    KEYS = ['dtlz2', 'tnk']
    FUNCTIONS = {'dtlz2': evaluate_DTLZ2, 'tnk': evaluate_TNK}
    VOCS = {'dtlz2': dtlz2_vocs, 'tnk': tnk_vocs}
    RPS = {'dtlz2': dtlz2_reference_point, 'tnk': tnk_reference_point}
    REPEATS = {'dtlz2': 10, 'tnk': 5}
    N_STEPS = 4

if __name__ == "__main__":
    bmobo = BenchMOBO()
    bmobo.run()
