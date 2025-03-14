import numpy as np
import pytest
from pydantic import ValidationError

from xopt import VOCS, Xopt
from xopt.errors import SeqGeneratorError
from xopt.generators.sequential.rcds import RCDSGenerator
from xopt.resources.testing import TEST_VOCS_BASE


def f_RCDS_minimize(input_dict):
    p = []
    for i in range(2):
        p.append(input_dict[f"p{i}"])

    obj = np.linalg.norm(p)
    outcome_dict = {"f": obj}

    return outcome_dict


class TestRCDSGenerator:
    def test_rcds_generate_multiple_points(self):
        gen = RCDSGenerator(vocs=TEST_VOCS_BASE)

        # Try to generate multiple samples
        with pytest.raises(SeqGeneratorError):
            gen.generate(2)

    def test_rcds_options(self):
        gen = RCDSGenerator(vocs=TEST_VOCS_BASE)

        with pytest.raises(ValidationError):
            gen.step = 0

        with pytest.raises(ValidationError):
            gen.tol = 0

    def test_rcds_yaml(self):
        YAML = """
        max_evaluations: 100
        generator:
            name: rcds
            init_mat: null
            noise: 0.00001
            step: 0.01
            tol: 0.00001
        evaluator:
            function: xopt.resources.test_functions.tnk.evaluate_TNK
        vocs:
            variables:
                x1: [0, 1]
                x2: [0, 1]
            objectives:
                y1: MINIMIZE
        """
        X = Xopt.from_yaml(YAML)
        X.random_evaluate(1)

        # test running multiple steps
        for i in range(10):
            X.step()

        assert X.generator.is_active
        assert X.generator._last_candidate is not None

        X.generator.reset()
        assert not X.generator.is_active

    def test_rcds_convergence(self):
        def eval_f(x):
            return {"y1": np.sum([x**2 for x in x.values()])}

        variables = {f"x{i}": [-5, 5] for i in range(10)}
        objectives = {"y1": "MINIMIZE"}
        vocs = VOCS(variables=variables, objectives=objectives)

        config = {
            "generator": {
                "name": "rcds",
            },
            "evaluator": {"function": eval_f},
            "vocs": vocs,
        }
        X = Xopt.from_dict(config)
        X.random_evaluate(1)
        for i in range(1000):
            X.step()
        print(X.data["y1"])

        idx, best, _ = X.vocs.select_best(X.data)
        xbest = X.vocs.variable_data(X.data.loc[idx, :]).to_numpy().flatten()
        assert best[0] >= 0.0
        assert best[0] <= 0.001
