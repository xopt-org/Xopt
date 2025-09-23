import logging
from copy import deepcopy

import torch

from xopt import VOCS
from xopt.generators.bayesian.models.standard import (
    BatchedModelConstructor,
    StandardModelConstructor,
)
from xopt.resources.benchmarking import BenchFunction
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA

logging.basicConfig(level=logging.DEBUG)


def generate_vocs(n_vars=5, n_obj=2, n_constr=2):
    variables = {f"x{i}": [0.0, 1.0] for i in range(n_vars)}
    objectives = {f"f{i}": "MAXIMIZE" for i in range(n_obj)}
    constraints = {f"c{i}": ["LESS_THAN", 0.5] for i in range(n_constr)}
    return VOCS(variables=variables, objectives=objectives, constraints=constraints)


def generate_data(vocs, n=100):
    import numpy as np
    import pandas as pd

    data = {}
    for var in vocs.variables:
        data[var] = np.random.rand(n)
    for i, obj in enumerate(vocs.objectives):
        data[obj] = np.linspace(0.0, 1.0, n) * (i + 1)
    for i, constr in enumerate(vocs.constraints):
        data[constr] = np.random.rand(n)
    return pd.DataFrame(data)


def test_model_batched():
    torch.set_num_threads(1)
    test_vocs = generate_vocs(n_vars=10, n_obj=5, n_constr=3)
    test_data = generate_data(vocs=test_vocs)

    def build_batched():
        gp_constructor = BatchedModelConstructor()
        model = gp_constructor.build_model_from_vocs(test_vocs, test_data)
        return model

    def build_standard():
        gp_constructor = StandardModelConstructor()
        model = gp_constructor.build_model_from_vocs(test_vocs, test_data)
        return model

    bench = BenchFunction()
    bench.add(build_batched)
    bench.add(build_standard)
    result1 = bench.run(min_rounds=10, warmup=True)
    print(result1)
