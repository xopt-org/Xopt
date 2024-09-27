import numpy as np
import pytest
from pydantic import ValidationError

from xopt import Xopt
from xopt.generators.rcds.rcds import RCDSGenerator
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
        with pytest.raises(NotImplementedError):
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
            x0: null
            init_mat: null
            noise: 0.00001
            step: 0.01
            tol: 0.00001
        evaluator:
            function: xopt.resources.test_functions.tnk.evaluate_TNK
        vocs:
            variables:
                p0: [0, 1]
                p1: [0, 1]
            objectives:
                f: MINIMIZE
        """
        X = Xopt.from_yaml(YAML)
        gen = X.generator
        gen.generate(1)
