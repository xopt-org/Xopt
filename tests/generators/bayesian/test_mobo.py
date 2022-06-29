from copy import deepcopy

import numpy as np
import pytest
import yaml

from xopt.evaluator import Evaluator
from xopt.base import Xopt
from xopt.generators.bayesian.mobo import MOBOGenerator
from xopt.resources.test_functions.tnk import evaluate_TNK, tnk_vocs
from xopt.resources.testing import TEST_VOCS_BASE


class TestBayesianExplorationGenerator:
    def test_init(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.objectives.update({"y2": "MINIMIZE"})
        gen = MOBOGenerator(vocs)

        print(f"\n{gen.options.dict()}")

    def test_script(self):
        evaluator = Evaluator(function=evaluate_TNK)

        # test check options
        bad_options = deepcopy(MOBOGenerator.default_options())
        bad_options.acq.proximal_lengthscales = [1.0, 1.0]

        bad_options2 = deepcopy(MOBOGenerator.default_options())
        bad_options2.optim.raw_samples = 1
        bad_options2.acq.monte_carlo_samples = 1
        bad_options2.acq.proximal_lengthscales = [1.0, 1.0, 1.0]

        for ele in [bad_options, bad_options2]:
            with pytest.raises(ValueError):
                MOBOGenerator(tnk_vocs, ele)

        base_options = deepcopy(MOBOGenerator.default_options())
        base_options.optim.raw_samples = 2
        base_options.acq.monte_carlo_samples = 2

        proximal_biasing = deepcopy(base_options)
        proximal_biasing.optim.num_restarts = 1  # required
        proximal_biasing.acq.proximal_lengthscales = [1.0, 1.0]

        proximal_biasing2 = deepcopy(base_options)
        proximal_biasing2.optim.num_restarts = 1  # required
        proximal_biasing2.acq.proximal_lengthscales = np.array([1.0, 1.0])

        for ele in [base_options, proximal_biasing, proximal_biasing2]:
            generator = MOBOGenerator(tnk_vocs, ele)
            X = Xopt(generator=generator, evaluator=evaluator, vocs=tnk_vocs)
            X.step()
            X.step()

    def test_yaml(self):
        YAML = """
        xopt: {}
        generator:
            name: mobo
            n_initial: 5
            optim:
                num_restarts: 1
                raw_samples: 2
            acq:
                proximal_lengthscales: [1.5, 1.5]

        evaluator:
            function: xopt.resources.test_functions.tnk.evaluate_TNK

        vocs:
            variables:
                x1: [0, 3.14159]
                x2: [0, 3.14159]
            objectives: {y1: MINIMIZE, y2: MINIMIZE}
            constraints:
                c1: [GREATER_THAN, 0]
                c2: [LESS_THAN, 0.5]
        """
        X = Xopt(config=yaml.safe_load(YAML))
        X.step()
        X.step()
