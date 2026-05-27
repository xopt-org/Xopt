import json

import pytest

from xopt import Xopt
from xopt.errors import SeqGeneratorError
from xopt.generators.sequential.scipy import ScipyGenerator


def sphere(input_dict):
    return {"y": float(sum(v**2 for v in input_dict.values()))}


class TestScipyGenerator:
    def test_scipy_generate_multiple_points(self):
        YAML = """
        generator:
            name: scipy
            method: Powell
            initial_point: {x0: 0.5, x1: -0.5}
            vocs:
                variables:
                    x0: [-5, 5]
                    x1: [-5, 5]
                objectives: {y: MINIMIZE}
        evaluator:
            function: xopt.tests.generators.sequential.test_scipy.sphere
        """
        X = Xopt.from_yaml(YAML)
        with pytest.raises(SeqGeneratorError):
            X.generator.generate(2)

    def test_scipy_generate_and_restart(self):
        YAML = """
        generator:
            name: scipy
            method: Powell
            initial_point: {x0: 1.2, x1: -1.1}
            options:
                maxiter: 200
            vocs:
                variables:
                    x0: [-5, 5]
                    x1: [-5, 5]
                objectives: {y: MINIMIZE}
        evaluator:
            function: xopt.tests.generators.sequential.test_scipy.sphere
        """
        X = Xopt.from_yaml(YAML)

        for _ in range(8):
            X.step()

        assert len(X.data) == 8

        state = X.json()
        X2 = Xopt.model_validate(json.loads(state))
        X2.step()

        assert len(X2.data) == 9

    def test_scipy_generator_direct(self):
        YAML = """
        generator:
            name: scipy
            method: Powell
            initial_point: {x0: 1.0, x1: 1.0}
            vocs:
                variables:
                    x0: [-5, 5]
                    x1: [-5, 5]
                objectives: {y: MINIMIZE}
        evaluator:
            function: xopt.tests.generators.sequential.test_scipy.sphere
        """
        X = Xopt.from_yaml(YAML)
        gen: ScipyGenerator = X.generator

        first = gen.generate(1)
        assert len(first) == 1
        assert set(first[0].keys()) == {"x0", "x1"}
