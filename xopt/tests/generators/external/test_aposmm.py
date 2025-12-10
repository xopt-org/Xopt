from libensemble.gen_classes.aposmm import APOSMM
import numpy as np
from xopt import Xopt
from xopt import Evaluator
from gest_api.vocs import VOCS
from copy import deepcopy
import pytest

def six_hump_camel_func(x):
    """Six-Hump Camel function definition"""
    x1 = x[0]
    x2 = x[1]
    term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2

    return term1 + term2 + term3

def evaluator_function(inputs):
    return {"f": six_hump_camel_func([inputs["x0"], inputs["x1"]])}


@pytest.fixture
def evaluator():
    return Evaluator(function=evaluator_function)

@pytest.fixture
def vocs():
    return VOCS(
        variables={
            "x0": [-2.0,2.0],
            "x1": [-1.0,1.0],
            "x0_on_cube": [0.0,1.0],
            "x1_on_cube": [0.0,1.0],
            },
        objectives={"f": "MINIMIZE"},
        )

@pytest.fixture
def mapping():
    return {
            "x": ["x0", "x1"],
            "x_on_cube": ["x0_on_cube", "x1_on_cube"],
        }

class TestXoptPlusAPOSMM:

    def test_init_with_aposmm(self, vocs, evaluator, mapping):
        
        gen = APOSMM(vocs=vocs, max_active_runs=1, initial_sample_size=40, variables_mapping=mapping)
        x = Xopt(generator=gen, evaluator=evaluator, vocs=vocs)
        import ipdb; ipdb.set_trace()
        print(';liksdhfoijuasdhdfoiusah')