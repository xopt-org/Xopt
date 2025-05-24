import importlib
import logging
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from botorch.utils.multi_objective import Hypervolume, is_non_dominated
from botorch.utils.multi_objective.box_decompositions import (
    DominatedPartitioning,
    FastNondominatedPartitioning,
)
from deap.tools._hypervolume.pyhv import _HyperVolume

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


have_pymoo = False
try:
    import importlib.util

    if importlib.util.find_spec("pymoo") is not None:
        have_pymoo = True
except ImportError:
    logger.warning("pymoo not installed, skipping pymoo benchmarks")
    pass


class BenchHV:
    def __init__(self, n_obj_list, it=20, n_array=None):
        # self.n_var_list = n_var_list or [10, 20]
        self.n_obj_list = n_obj_list or [2, 3]
        self.it = it
        self.n_array = (
            n_array
            if n_array is not None
            else np.linspace(10, 5000, 20).astype(np.int32)
        )

    def run(self):
        results = []
        for n_obj in self.n_obj_list:
            for npoints in self.n_array:
                print(f"Running {n_obj} objectives with {npoints} points")
                r = self.test_hv_performance(n_obj=n_obj, it=self.it, n_points=npoints)
                results.append(r)
        return results

    def test_hv_performance(self, n_obj=2, it=20, n_points=100):
        objectives = {f"y{i}": "MAXIMIZE" for i in range(n_obj)}
        # variables = {f'x{i}': [0.0, 1.0] for i in range(n_var)}

        Y = np.random.randn(n_points, n_obj) - np.ones((n_points, n_obj)) / 3
        Y_torch = torch.from_numpy(Y)
        ref_point = {x: 0.0 for x in objectives.keys()}
        ref_point_numpy = np.array(list(ref_point.values()))
        ref_point_torch = torch.from_numpy(ref_point_numpy)

        def compute_hv_pymoo(Y: np.ndarray, ref_point):
            from pymoo.indicators.hv import HV

            hv = HV(ref_point=ref_point)
            volume = float(hv(Y))
            return volume

        def compute_hv_botorch(Y: torch.Tensor, ref_point):
            hvo = Hypervolume(ref_point=ref_point)
            pareto_mask = is_non_dominated(Y)
            pareto_y = Y[pareto_mask]
            volume = float(hvo.compute(pareto_y))
            return volume

        def compute_hv_botorch_partitioning(Y: torch.Tensor, ref_point):
            bd = DominatedPartitioning(ref_point=ref_point, Y=Y)
            volume = float(bd.compute_hypervolume().item())
            return volume

        def compute_hv_botorch_fndpartitioning(Y: torch.Tensor, ref_point):
            bd = FastNondominatedPartitioning(ref_point=ref_point, Y=Y)
            volume = float(bd.compute_hypervolume().item())
            return volume

        # this is not working
        def compute_hv_deap(Y: np.ndarray, ref_point):
            hv = _HyperVolume(ref_point)
            volume = float(hv.compute(Y))
            return volume

        def compute_pf_botorch(Y: torch.Tensor):
            pareto_mask = is_non_dominated(Y, deduplicate=False)
            pareto_y = Y[pareto_mask]
            return pareto_y

        def compute_pf_botorch_partitioning(Y: torch.Tensor, ref_point):
            bd = DominatedPartitioning(ref_point=ref_point, Y=Y)
            return bd.pareto_Y

        def compute_pf_botorch_fndpartitioning(Y: torch.Tensor, ref_point):
            bd = FastNondominatedPartitioning(ref_point=ref_point, Y=Y)
            return bd.pareto_Y

        def accumulate(f, it, **kwargs):
            vals = []
            t1 = time.perf_counter()
            for i in range(it):
                vals.append(f(**kwargs))
            t2 = time.perf_counter()
            return vals, (t2 - t1) / it

        if have_pymoo:
            v_hv_pymoo, t_pymoo = accumulate(
                compute_hv_pymoo, it, Y=Y, ref_point=ref_point_numpy
            )
        v_hv_botorch, t_botorch_hypervolume = accumulate(
            compute_hv_botorch, it, Y=-Y_torch, ref_point=-ref_point_torch
        )
        v_hv_botorch_gpu, t_botorch_hypervolume_gpu = accumulate(
            compute_hv_botorch,
            it,
            Y=-Y_torch.to("cuda:0"),
            ref_point=-ref_point_torch.to("cuda:0"),
        )
        v_hv_botorch_partitioning, t_botorch_partitioning = accumulate(
            compute_hv_botorch_partitioning, it, Y=-Y_torch, ref_point=-ref_point_torch
        )
        v_hv_botorch_partitioning_gpu, t_botorch_partitioning_gpu = accumulate(
            compute_hv_botorch_partitioning,
            it,
            Y=-Y_torch.to("cuda:0"),
            ref_point=-ref_point_torch.to("cuda:0"),
        )
        v_hv_botorch_fndpartitioning, t_botorch_partitioning_fnd = accumulate(
            compute_hv_botorch_fndpartitioning,
            it,
            Y=-Y_torch,
            ref_point=-ref_point_torch,
        )
        # v_hv_deap, t_deap = accumulate(compute_hv_deap, it, Y=Y, ref_point=ref_point_numpy)

        v_pf_botorch, t_botorch_pf = accumulate(compute_pf_botorch, it, Y=-Y_torch)
        v_pf_botorch_partitioning, t_botorch_partitioning_pf = accumulate(
            compute_pf_botorch_partitioning, it, Y=-Y_torch, ref_point=-ref_point_torch
        )
        # v_pf_botorch_fndpartitioning, t_botorch_partitioning_fnd = accumulate(compute_pf_botorch_fndpartitioning, it,
        #                                                                       Y=-Y_torch,
        #                                                                       ref_point=-ref_point_torch)

        if have_pymoo:
            assert np.allclose(v_hv_pymoo, v_hv_botorch), (
                f"{v_hv_pymoo} != {v_hv_botorch}"
            )
        assert np.allclose(v_hv_pymoo, v_hv_botorch_gpu), (
            f"{v_hv_pymoo} != {v_hv_botorch_gpu}"
        )
        assert np.allclose(v_hv_pymoo, v_hv_botorch_partitioning), (
            f"{v_hv_pymoo} != {v_hv_botorch_partitioning}"
        )
        assert np.allclose(v_hv_pymoo, v_hv_botorch_partitioning_gpu), (
            f"{v_hv_pymoo} != {v_hv_botorch_partitioning_gpu}"
        )
        assert np.allclose(v_hv_pymoo, v_hv_botorch_fndpartitioning), (
            f"{v_hv_pymoo} != {v_hv_botorch_fndpartitioning}"
        )
        # assert np.allclose(v_hv_pymoo, v_hv_deap), f'{v_hv_pymoo} != {v_hv_deap}'

        # print(v_pf_botorch[0])
        # print(v_pf_botorch_partitioning[0])
        # print(v_pf_botorch_fndpartitioning[0])
        pf1 = np.array(v_pf_botorch[0])
        pf2 = np.array(v_pf_botorch_partitioning[0])
        # non-dominated return more points vs partitioning
        r = True
        try:
            # check if each row in pf2 is in pf1
            for p in pf2:
                if not np.any(np.all(pf1 == p, axis=1)):
                    print(f"Not found {p} in {pf1}")
                    r = False
                    break
        except Exception:
            raise ValueError(f"Failed to compare {pf1} and {pf2}")

        if not r:
            raise ValueError(f"{pf1} != {pf2}")

        r = {
            "t_botorch_hypervolume": t_botorch_hypervolume,
            "t_botorch_hypervolume_gpu": t_botorch_hypervolume_gpu,
            "t_botorch_partitioning": t_botorch_partitioning,
            "t_botorch_partitioning_gpu": t_botorch_partitioning_gpu,
            "t_botorch_partitioning_fnd": t_botorch_partitioning_fnd,
            "t_botorch_pf": t_botorch_pf,
            "t_botorch_partitioning_pf": t_botorch_partitioning_pf,
            #'t_deap': t_deap,
            "n_obj": n_obj,
            "n_points": n_points,
        }
        if have_pymoo:
            r["t_pymoo"] = t_pymoo
        return r


def time_call(f: Callable, n: int = 1) -> tuple[list[float], list[Any]]:
    """
    Time a function call
    """
    times = []
    results = []
    for _ in range(n):
        start = time.perf_counter()
        v = f()
        end = time.perf_counter()
        times.append(end - start)
        results.append(v)
    return times, results
