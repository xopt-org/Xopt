import json

import numpy as np
import pytest
from scipy.optimize import minimize

from xopt import Xopt
from xopt.errors import SeqGeneratorError
from xopt.generators.sequential.cobyqa import COBYQAGenerator


def _has_scipy_cobyqa() -> bool:
    def objective(x):
        return float(np.sum(np.array(x) ** 2))

    try:
        minimize(objective, [1.0], method="cobyqa", bounds=[(-2.0, 2.0)])
    except ValueError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _has_scipy_cobyqa(), reason="scipy COBYQA is not available in this environment"
)


def sphere(input_dict):
    return {"y": float(sum(v**2 for v in input_dict.values()))}


class TestCOBYQAGenerator:
    def test_cobyqa_generate_multiple_points(self):
        YAML = """
        generator:
            name: cobyqa
            initial_point: {x0: 0.5, x1: -0.5}
            vocs:
                variables:
                    x0: [-5, 5]
                    x1: [-5, 5]
                objectives: {y: MINIMIZE}
        evaluator:
            function: xopt.tests.generators.sequential.test_cobyqa.sphere
        """
        X = Xopt.from_yaml(YAML)
        with pytest.raises(SeqGeneratorError):
            X.generator.generate(2)

    def test_cobyqa_generate_and_restart(self):
        YAML = """
        generator:
            name: cobyqa
            initial_point: {x0: 1.2, x1: -1.1}
            options:
                maxiter: 200
            vocs:
                variables:
                    x0: [-5, 5]
                    x1: [-5, 5]
                objectives: {y: MINIMIZE}
        evaluator:
            function: xopt.tests.generators.sequential.test_cobyqa.sphere
        """
        X = Xopt.from_yaml(YAML)

        for _ in range(8):
            X.step()

        assert len(X.data) == 8

        state = X.json()
        X2 = Xopt.model_validate(json.loads(state))
        X2.step()

        assert len(X2.data) == 9

    def test_cobyqa_generator_direct(self):
        YAML = """
        generator:
            name: cobyqa
            initial_point: {x0: 1.0, x1: 1.0}
            vocs:
                variables:
                    x0: [-5, 5]
                    x1: [-5, 5]
                objectives: {y: MINIMIZE}
        evaluator:
            function: xopt.tests.generators.sequential.test_cobyqa.sphere
        """
        X = Xopt.from_yaml(YAML)
        gen: COBYQAGenerator = X.generator

        first = gen.generate(1)
        assert len(first) == 1
        assert set(first[0].keys()) == {"x0", "x1"}
