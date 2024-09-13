import logging
import time

import pandas as pd

from xopt import Evaluator, Xopt
from xopt.generators.bayesian import MOBOGenerator
from xopt.resources.test_functions.multi_objective import DTLZ2, LinearMO, QuadraticMO
from xopt.resources.test_functions.tnk import (
    evaluate_TNK,
    tnk_reference_point,
    tnk_vocs,
)

logger = logging.getLogger(__name__)

quad = QuadraticMO()
lin = LinearMO()
dtlz2_3d = DTLZ2()
dtlz2_20d = DTLZ2(n_var=20)


# Can move into tests, but annoying to run
# benchmark = pytest.mark.skipif('--run-bench' not in sys.argv, reason="No benchmarking requested")
class BenchMOBO:
    """
    Benchmark class for running various configurations either directly or through the pytest interface
    """

    KEYS = ["quad", "linear", "dtlz2_d3", "dtlz2_d20"]
    FUNCTIONS = {
        "dtlz2_d3": dtlz2_3d.evaluate_dict,
        "dtlz2_d20": dtlz2_20d.evaluate_dict,
        "linear": lin.evaluate_dict,
        "tnk": evaluate_TNK,
        "quad": quad.evaluate_dict,
    }
    VOCS = {
        "dtlz2_d3": dtlz2_3d.vocs,
        "dtlz2_d20": dtlz2_20d.vocs,
        "linear": lin.vocs,
        "tnk": tnk_vocs,
        "quad": quad.vocs,
    }
    RPS = {
        "dtlz2_d3": dtlz2_3d.ref_point_dict,
        "dtlz2_d20": dtlz2_20d.ref_point_dict,
        "linear": lin.ref_point_dict,
        "tnk": tnk_reference_point,
        "quad": quad.ref_point_dict,
    }
    REPEATS = {"dtlz2_d3": 1, "dtlz2_d20": 1, "linear": 1, "tnk": 1, "quad": 1}
    N_STEPS = 5

    OPTS = [
        dict(n_monte_carlo_samples=8, log_transform_acquisition_function=False),
        dict(n_monte_carlo_samples=32, log_transform_acquisition_function=False),
        dict(n_monte_carlo_samples=128, log_transform_acquisition_function=False),
        dict(n_monte_carlo_samples=8, log_transform_acquisition_function=True),
        dict(n_monte_carlo_samples=32, log_transform_acquisition_function=True),
        dict(n_monte_carlo_samples=128, log_transform_acquisition_function=True),
    ]

    def __init__(self, df):
        self.df = df

    def run_opt(self, gen, evaluator, n_evals):
        """Run xopt with specified generator and evaluator"""
        X = Xopt(generator=gen, evaluator=evaluator, vocs=gen.vocs)
        X.random_evaluate(2)
        X.data.loc[:, "gen_time"] = 0.0
        X.data.loc[:, "hv"] = 0.0
        for i in range(n_evals):
            t1 = time.perf_counter()
            X.step()
            t2 = time.perf_counter()
            X.data.iloc[-1, X.data.columns.get_loc("gen_time")] = t2 - t1
            X.data.iloc[-1, X.data.columns.get_loc("hv")] = (
                X.generator.calculate_hypervolume()
            )
        return X

    @classmethod
    def crate_parameter_table(cls):
        """Create a table of generator parameters to benchmark"""
        rows = []
        for k in cls.KEYS:
            for i, opts in enumerate(cls.OPTS):
                rows.append(
                    {
                        "k": f"{k}_{i}",
                        "fname": k,
                        "opts": opts,
                        "rp": cls.RPS[k],
                        "vocs": cls.VOCS[k],
                    }
                )

        return pd.DataFrame(rows)

    def run(self, row):
        evaluator = Evaluator(function=self.FUNCTIONS[self.df.loc[row, "fname"]])
        gen = MOBOGenerator(
            vocs=self.df.loc[row, "vocs"],
            reference_point=self.df.loc[row, "rp"],
            **self.df.loc[row, "opts"],
        )
        X = self.run_opt(gen, evaluator, self.N_STEPS)

        outputs = {
            "k": self.df.loc[row, "k"],
            "fname": self.df.loc[row, "fname"],
            **self.df.loc[row, "opts"],
        }
        outputs["t"] = X.data.gen_time.sum()
        outputs["hv25"] = X.data.loc[1 * self.N_STEPS // 4, "hv"]
        outputs["hv50"] = X.data.loc[2 * self.N_STEPS // 4, "hv"]
        outputs["hv75"] = X.data.loc[3 * self.N_STEPS // 4, "hv"]
        outputs["hvf"] = X.generator.calculate_hypervolume()

        return outputs
