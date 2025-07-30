from copy import deepcopy
import numpy as np
import pytest
from pydantic import ValidationError

from xopt import Evaluator, VOCS, Xopt
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


def eval_f_linear_pos(x):
    return {"y1": np.sum([x**2 for x in x.values()])}


def eval_f_linear_neg(x):
    return {"y1": -np.sum([x**2 for x in x.values()])}


def eval_f_linear_offset(x):  # offset the optimal solution
    return {"y1": np.sum([(x - 2) ** 2 for x in x.values()])}


class TestRCDSGenerator:
    def test_rcds_generate_multiple_points(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.constraints = {}

        gen = RCDSGenerator(vocs=test_vocs)

        # Try to generate multiple samples
        with pytest.raises(SeqGeneratorError):
            gen.generate(2)

    def test_rcds_options(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.constraints = {}

        gen = RCDSGenerator(vocs=test_vocs)

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

    @pytest.mark.parametrize(
        "fun, obj, x_opt, max_iter",
        [
            (eval_f_linear_pos, "MINIMIZE", np.zeros(10), 3000),
            (eval_f_linear_neg, "MAXIMIZE", np.zeros(10), 3000),
            (eval_f_linear_offset, "MINIMIZE", 2 * np.ones(10), 3000),
        ],
    )
    def test_rcds_convergence(self, fun, obj, x_opt, max_iter):
        variables = {f"x{i}": [-5, 5] for i in range(len(x_opt))}
        objectives = {"y1": obj}
        vocs = VOCS(variables=variables, objectives=objectives)
        generator = RCDSGenerator(step=0.01, noise=0.00001, vocs=vocs)
        evaluator = Evaluator(function=fun)
        X = Xopt(vocs=vocs, evaluator=evaluator, generator=generator)

        if x_opt.sum():  # if the optimal solution is not 0
            X.evaluate_data({f"x{i}": 1.2 for i in range(len(x_opt))})
        else:
            X.random_evaluate(1)
        for i in range(max_iter):
            X.step()

        idx, best, _ = X.vocs.select_best(X.data)
        xbest = X.vocs.variable_data(X.data.loc[idx, :]).to_numpy().flatten()
        if obj == "MINIMIZE":
            assert best[0] >= 0.0
            assert best[0] <= 0.001
        else:
            assert best[0] <= 0.0
            assert best[0] >= -0.001
        assert np.allclose(xbest, x_opt, rtol=0, atol=1e-3)
