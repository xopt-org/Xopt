import math
from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from xopt import Evaluator, VOCS, Xopt
from xopt.generators import UpperConfidenceBoundGenerator
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.options import OptimizationOptions
from xopt.generators.bayesian.turbo import TurboController
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA


class TestTurbo(TestCase):
    def test_turbo_init(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.variables = {"x1": [0, 1]}

        state = TurboController(test_vocs)
        assert state.dim == 1
        assert state.failure_tolerance == 2
        assert state.success_tolerance == 2

    @patch.multiple(BayesianGenerator, __abstractmethods__=set())
    def test_get_trust_region(self):
        # test in 1D
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.variables = {"x1": [0, 1]}

        gen = BayesianGenerator(vocs=test_vocs)
        gen.add_data(TEST_VOCS_DATA)
        model = gen.train_model()

        turbo_state = TurboController(gen.vocs)
        turbo_state.update_state(gen.data)
        tr = turbo_state.get_trust_region(model)
        assert tr[0].numpy() >= test_vocs.bounds[0]
        assert tr[1].numpy() <= test_vocs.bounds[1]

        # test in 2D
        test_vocs = deepcopy(TEST_VOCS_BASE)
        gen = BayesianGenerator(vocs=test_vocs)
        gen.add_data(TEST_VOCS_DATA)
        model = gen.train_model()

        turbo_state = TurboController(gen.vocs)
        turbo_state.update_state(gen.data)
        tr = turbo_state.get_trust_region(model)

        assert np.all(tr[0].numpy() >= test_vocs.bounds[0])
        assert np.all(tr[1].numpy() <= test_vocs.bounds[1])

        with pytest.raises(RuntimeError):
            turbo_state = TurboController(gen.vocs)
            turbo_state.get_trust_region(model)

    def test_set_best_point(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)

        turbo_state = TurboController(test_vocs)
        turbo_state.update_state(TEST_VOCS_DATA)
        assert (
            turbo_state.best_value == TEST_VOCS_DATA.min()[test_vocs.objective_names[0]]
        )

    def test_in_generator(self):
        vocs = VOCS(
            variables={"x": [0, 2 * math.pi]},
            objectives={"f": "MINIMIZE"},
        )

        import numpy as np

        def sin_function(input_dict):
            x = input_dict["x"]
            return {"f": -10 * np.exp(-((x - np.pi) ** 2) / 0.01) + 0.5 * np.sin(5 * x)}

        evaluator = Evaluator(function=sin_function)
        generator = UpperConfidenceBoundGenerator(vocs=vocs,
                                                  turbo_controller="controller")
        X = Xopt(evaluator=evaluator, generator=generator, vocs=vocs)

        X.evaluate_data(pd.DataFrame({"x": [3.0, 1.75, 2.0]}))

        # determine trust region from gathered data
        generator.train_model()
        generator.turbo_controller.update_state(generator.data)
        generator.turbo_controller.get_trust_region(generator.model)
