import logging
import time

import torch

from xopt.generators.bayesian import ExpectedImprovementGenerator
from xopt.resources.benchmarking import BenchDispatcher, generate_data, generate_vocs
from xopt.generators.bayesian.models.standard import (
    BatchedModelConstructor,
)
import threadpoolctl

logging.basicConfig(level=logging.DEBUG)


def bench_build_standard_kwargs():
    test_vocs = generate_vocs(n_vars=5, n_obj=1, n_constr=2)
    test_data = generate_data(vocs=test_vocs, n=500)
    return test_vocs, test_data


def preamble():
    torch.set_num_threads(1)
    threadpoolctl.threadpool_limits(limits=1, user_api="blas")
    threadpoolctl.threadpool_limits(limits=1, user_api="openmp")
    torch.cuda.synchronize()


def timeit(func, *args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return elapsed_time, result


@BenchDispatcher.register_decorator(preamble=preamble)
@BenchDispatcher.register_defaults(
    ["vocs", "data"], lambda: bench_build_standard_kwargs()
)
def bench_generate_standard(vocs, data, device):
    device = torch.device(device)
    generator_batched = ExpectedImprovementGenerator(
        vocs=vocs, use_cuda=True if device.type == "cuda" else False
    )
    generator_batched.add_data(data)
    t, r = timeit(generator_batched.train_model)
    return t, r


@BenchDispatcher.register_decorator(preamble=preamble)
@BenchDispatcher.register_defaults(
    ["vocs", "data"], lambda: bench_build_standard_kwargs()
)
def bench_generate_batched(vocs, data, device):
    device = torch.device(device)
    generator_batched = ExpectedImprovementGenerator(
        vocs=vocs,
        gp_constructor=BatchedModelConstructor(),
        use_cuda=True if device.type == "cuda" else False,
    )
    generator_batched.add_data(data)
    t, r = timeit(generator_batched.train_model)
    return t, r
