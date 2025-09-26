import logging

import torch

from xopt.resources.benchmarking import BenchDispatcher, generate_data, generate_vocs
from xopt.generators.bayesian.models.standard import (
    BatchedModelConstructor,
    StandardModelConstructor,
)

logging.basicConfig(level=logging.DEBUG)


def bench_build_standard_kwargs():
    test_vocs = generate_vocs(n_vars=10, n_obj=5, n_constr=2)
    test_data = generate_data(vocs=test_vocs, n=100)
    return test_vocs, test_data


def preamble_build_model():
    # import numpy as np
    # np.show_runtime()
    torch.set_num_threads(1)
    # botorch already uses same method internally, but force globally for other libs
    import threadpoolctl
    from pprint import pprint

    threadpoolctl.threadpool_limits(limits=1, user_api="blas")
    threadpoolctl.threadpool_limits(limits=1, user_api="openmp")
    pprint(threadpoolctl.threadpool_info())


@BenchDispatcher.register_decorator(preamble=preamble_build_model)
@BenchDispatcher.register_defaults(
    ["vocs", "data"], lambda: bench_build_standard_kwargs()
)
def bench_build_standard(vocs, data, device="cpu"):
    device = torch.device(device)
    torch.cuda.empty_cache()
    gp_constructor = StandardModelConstructor()
    model = gp_constructor.build_model_from_vocs(vocs, data, device=device)
    return model


@BenchDispatcher.register_decorator(preamble=preamble_build_model)
@BenchDispatcher.register_defaults(
    ["vocs", "data"], lambda: bench_build_standard_kwargs()
)
def bench_build_batched(vocs, data, device="cpu"):
    device = torch.device(device)
    torch.cuda.empty_cache()
    gp_constructor = BatchedModelConstructor()
    model = gp_constructor.build_model_from_vocs(vocs, data, device=device)
    return model
