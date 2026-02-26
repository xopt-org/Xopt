import numpy as np
from libensemble.gen_classes.aposmm import APOSMM
from xopt import Xopt
from xopt import Evaluator
from xopt.stopping_conditions import MaxEvaluationsCondition
from gest_api.vocs import VOCS
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
def max_evaluations():
    return MaxEvaluationsCondition(max_evaluations=45)


@pytest.fixture
def vocs():
    return VOCS(
        variables={
            "x0": [-2.0, 2.0],
            "x1": [-1.0, 1.0],
            "x0_on_cube": [0.0, 1.0],
            "x1_on_cube": [0.0, 1.0],
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
    def test_init(self, vocs, evaluator, max_evaluations, mapping):
        gen = APOSMM(
            vocs=vocs,
            max_active_runs=1,
            initial_sample_size=40,
            variables_mapping=mapping,
        )
        Xopt(
            generator=gen,
            evaluator=evaluator,
            stopping_condition=max_evaluations,
        )

    def test_run(self, vocs, evaluator, max_evaluations, mapping):
        gen = APOSMM(
            vocs=vocs,
            max_active_runs=1,
            initial_sample_size=40,
            variables_mapping=mapping,
        )
        x = Xopt(
            generator=gen,
            evaluator=evaluator,
            stopping_condition=max_evaluations,
        )
        x.run()
        assert x.data.shape[0] == 45
        assert all(x.data["local_pt"][-5:]), (
            "last 5 points should be local_pt, as communicated by APOSMM"
        )

    def test_random_evaluate(self, vocs, evaluator, max_evaluations, mapping):
        gen = APOSMM(
            vocs=vocs,
            max_active_runs=1,
            initial_sample_size=40,
            variables_mapping=mapping,
        )
        x = Xopt(
            generator=gen,
            evaluator=evaluator,
            stopping_condition=max_evaluations,
        )
        # Generate proper x/x_on_cube pairs (cube = normalized to [0,1])
        rng = np.random.default_rng(seed=42)
        n = 40
        x0 = rng.uniform(-2.0, 2.0, n)
        x1 = rng.uniform(-1.0, 1.0, n)
        x0_on_cube = (x0 - (-2.0)) / (2.0 - (-2.0))
        x1_on_cube = (x1 - (-1.0)) / (1.0 - (-1.0))
        x.evaluate_data(
            {"x0": x0, "x1": x1, "x0_on_cube": x0_on_cube, "x1_on_cube": x1_on_cube}
        )
        x.step()
        x.generator.finalize()
        assert x.data.shape[0] == 41
        assert all(x.data["local_pt"][-1:]), (
            "last point should be local_pt, as communicated by APOSMM"
        )

    def test_step(self, vocs, evaluator, max_evaluations, mapping):
        gen = APOSMM(
            vocs=vocs,
            max_active_runs=1,
            initial_sample_size=40,
            variables_mapping=mapping,
        )
        x = Xopt(
            generator=gen,
            evaluator=evaluator,
            stopping_condition=max_evaluations,
        )
        for i in range(45):
            x.step()
        x.generator.finalize()
        assert x.data.shape[0] == 45
        assert all(x.data["local_pt"][-5:]), (
            "last 5 points should be local_pt, as communicated by APOSMM"
        )
