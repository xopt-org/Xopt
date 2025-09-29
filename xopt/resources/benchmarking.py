import cProfile
import logging
import pstats
import time
from functools import wraps
from typing import Any, Callable

import numpy as np
import pandas as pd

from xopt import Evaluator, VOCS, Xopt
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
                X.generator.get_pareto_front_and_hypervolume()[-1]
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
        outputs["hvf"] = X.generator.get_pareto_front_and_hypervolume()[-1]

        return outputs


class BenchFunction:
    def __init__(self):
        self.configs = []

    def add(self, f, args=None, kwargs=None, preamble=None):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        self.configs.append((f, args, kwargs, preamble))

    def run_config(
        self,
        min_time: float = 1.0,
        max_time: float = 600.0,
        min_rounds: int = 2,
        disable_gc: bool = False,
        warmup: bool = False,
        row: int = 0,
    ):
        f, args, kwargs, pre = self.configs[row]
        import gc

        try:
            if pre is not None:
                pre()
            if warmup:
                f(*args, **kwargs)
            if disable_gc:
                gc.disable()
            n = 0
            times = []
            total_time = 0.0
            while True:
                t1 = time.perf_counter()
                f(*args, **kwargs)
                t2 = time.perf_counter()
                t = t2 - t1
                times.append(t)
                total_time += t
                n += 1
                if total_time > max_time and n >= min_rounds:
                    break
                if n >= min_rounds and total_time > min_time:
                    break
                time.sleep(0.01)
            return {
                "f": f.__name__,
                "n_runs": n,
                "t_mean": total_time / n,
                "t_median": float(np.median(times)),
                "t_max": float(np.max(times)),
                "t_min": float(np.min(times)),
                "t_total": total_time,
                "stdev": float(np.std(times)),
                "args": args,
                "kwargs": kwargs,
            }
        finally:
            if disable_gc:
                gc.enable()

    def run(self, **kwargs):
        results = []
        for i in range(len(self.configs)):
            results.append(self.run_config(row=i, **kwargs))
        r = pd.DataFrame(results)
        rdisp = r.drop(columns=["args", "kwargs"])
        # for i in range(len(self.configs)):
        #     print("Results for function:", self.configs[i][0].__name__)
        #     print(rdisp.iloc[i])

        print("---Results---")
        print(rdisp.to_markdown())


class BenchDispatcher:
    benchmarks = {}
    arguments = {}
    verbose = False

    @staticmethod
    def register(func, name=None, preamble=None):
        if BenchDispatcher.verbose:
            print(f"Registering benchmark function {name or func.__name__}")
        if name is None:
            name = func.__name__
        BenchDispatcher.benchmarks[name] = [preamble, func]

    @staticmethod
    def register_decorator(name=None, preamble=None):
        def decorator(func):
            BenchDispatcher.register(func, name, preamble)
            return func

        return decorator

    @staticmethod
    def register_defaults(arg_names: list[str], arg_generator):
        def decorator(func):
            name = func.__name__
            if name not in BenchDispatcher.arguments:
                BenchDispatcher.arguments[name] = []
            for arg_name in arg_names:
                if arg_name in BenchDispatcher.arguments[name]:
                    raise ValueError(
                        f"Argument {arg_name} already registered for function {name}"
                    )
            if not callable(arg_generator):
                if not isinstance(arg_generator, dict):
                    raise ValueError("arg_generator must be a callable or dict")
            BenchDispatcher.arguments[name].append([arg_names, arg_generator])
            return func

        return decorator

    @staticmethod
    def get_kwargs(name):
        if name not in BenchDispatcher.arguments:
            return {}
        kwargs = {}
        for arg_names, v in BenchDispatcher.arguments[name]:
            print(f"Generating args {arg_names} for function {name} from {v}")
            if callable(v):
                v = v()
            for k2, v2 in zip(arg_names, v):
                if k2 in kwargs:
                    raise ValueError(f"Argument {k2} already set for function {name}")
                kwargs[k2] = v2
        return kwargs

    @staticmethod
    def get(name):
        if name not in BenchDispatcher.benchmarks:
            raise KeyError(
                f"Benchmark {name} not found, available: {list(BenchDispatcher.benchmarks.keys())}"
            )
        preamble, func = BenchDispatcher.benchmarks.get(name)
        return preamble, func


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


def profile_function(func, dump_stats: bool = False):
    """A decorator to profile a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Use an environment variable to turn profiling on/off
        profiler = cProfile.Profile()
        result = profiler.runctx("func(*args, **kwargs)", globals(), locals())
        if dump_stats:
            stats_file = f"{func.__name__}.prof"
            profiler.dump_stats(stats_file)
            print(f"Profiling stats saved to '{stats_file}'.")

        pstats.Stats(profiler).sort_stats("cumulative").print_stats(10)
        return result

    return wrapper


def generate_vocs(n_vars=5, n_obj=2, n_constr=2):
    variables = {f"x{i}": [0.0, 1.0] for i in range(n_vars)}
    objectives = {f"f{i}": "MAXIMIZE" for i in range(n_obj)}
    constraints = {f"c{i}": ["LESS_THAN", 0.5] for i in range(n_constr)}
    return VOCS(variables=variables, objectives=objectives, constraints=constraints)


def generate_data(vocs, n=100):
    data = {}
    for var in vocs.variables:
        data[var] = np.linspace(0.0, 1.0, num=n)
    for i, obj in enumerate(vocs.objectives):
        data[obj] = (np.linspace(0.0, 1.0, n) - 0.5) * (i + 1)
    for i, constr in enumerate(vocs.constraints):
        data[constr] = np.sin(np.linspace(0.0, 1.0, num=n) / (2 * np.pi) + (i + 1))
    return pd.DataFrame(data)
