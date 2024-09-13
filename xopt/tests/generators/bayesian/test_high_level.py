import os
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from xopt import Xopt
from xopt.generators.bayesian.upper_confidence_bound import (
    UpperConfidenceBoundGenerator,
)
from xopt.resources.testing import TEST_VOCS_BASE


class TestHighLevel:
    def test_ucb(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.constraints = {}
        ucb_gen = UpperConfidenceBoundGenerator(vocs=test_vocs)
        ucb_gen.beta = 0.0
        ucb_gen.n_monte_carlo_samples = 512
        # add data
        data = pd.DataFrame({"x1": [0.0, 1.0], "x2": [0.0, 1.0], "y1": [1.0, -10.0]})
        ucb_gen.add_data(data)
        model = ucb_gen.train_model(ucb_gen.data)
        acq = ucb_gen.get_acquisition(model)

        n = 10
        bounds = ucb_gen.vocs.bounds
        x = torch.linspace(*bounds.T[0], n)
        y = torch.linspace(*bounds.T[1], n)
        xx, yy = torch.meshgrid(x, y)
        pts = torch.hstack([ele.reshape(-1, 1) for ele in (xx, yy)]).double()

        with torch.no_grad():
            acq_val = acq(pts.unsqueeze(1))
            assert torch.allclose(
                acq_val.max(), torch.tensor(9.9955).double(), atol=0.1
            )

    def test_constrained_mobo(self):
        YAML = """
        generator:
            name: mobo
            reference_point: {y1: 1.5, y2: 1.5}
            numerical_optimizer:
                name: LBFGS
                n_restarts: 1

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
        X = Xopt.from_yaml(YAML)
        X.random_evaluate(3)  # generates random data
        X.step()  # actually evaluates mobo

    def test_mobo(self):
        YAML = """
            generator:
                name: mobo
                reference_point: {y1: 1.5, y2: 1.5}
                numerical_optimizer:
                    name: LBFGS
                    n_restarts: 2
            evaluator:
                function: xopt.resources.test_functions.tnk.evaluate_TNK
            vocs:
                variables:
                    x1: [0, 3.14159]
                    x2: [0, 3.14159]
                objectives: {y1: MINIMIZE, y2: MINIMIZE}
                constraints: {}
        """
        X = Xopt.from_yaml(YAML)
        X.random_evaluate(3)  # generates random data
        X.step()  # actually evaluates mobo

    def test_restart_torch_inline_serialization(self):
        YAML = """
                dump_file: dump_inline.yml
                serialize_torch: True
                serialize_inline: True

                generator:
                    name: mobo
                    reference_point: {y1: 1.5, y2: 1.5}
                    numerical_optimizer:
                        name: LBFGS
                        n_restarts: 1
                evaluator:
                    function: xopt.resources.test_functions.tnk.evaluate_TNK
                vocs:
                    variables:
                        x1: [0, 3.14159]
                        x2: [0, 3.14159]
                    objectives: {y1: MINIMIZE, y2: MINIMIZE}
                    constraints: {}
                """
        X = Xopt.from_yaml(YAML)
        X.random_evaluate(3)
        X.step()

        out = X.json()
        assert len(out) > 500

        assert not os.path.exists("generator_model.pt")
        config = yaml.safe_load(open("dump_inline.yml"))

        X2 = Xopt.model_validate(config)

        assert X2.generator.vocs.variable_names == ["x1", "x2"]
        assert X2.generator.numerical_optimizer.n_restarts == 1
        assert np.allclose(
            X2.generator.data[X2.vocs.all_names].to_numpy(),
            X.data[X.vocs.all_names].to_numpy(),
        )
        assert (
            X.generator.model.state_dict().__str__()
            == X2.generator.model.state_dict().__str__()
        )

        X2.step()

    def test_restart_torch_serialization(self):
        YAML = """
                dump_file: dump.yml
                serialize_torch: True

                generator:
                    name: mobo
                    reference_point: {y1: 1.5, y2: 1.5}
                    numerical_optimizer:
                        name: LBFGS
                        n_restarts: 1
                evaluator:
                    function: xopt.resources.test_functions.tnk.evaluate_TNK
                vocs:
                    variables:
                        x1: [0, 3.14159]
                        x2: [0, 3.14159]
                    objectives: {y1: MINIMIZE, y2: MINIMIZE}
                    constraints: {}
                """
        X = Xopt.from_yaml(YAML)
        X.random_evaluate(3)
        X.step()

        config = yaml.safe_load(open("dump.yml"))
        assert config["generator"]["model"] == "generator_model.pt"

        # test restart
        X2 = Xopt.model_validate(config)

        assert X2.generator.vocs.variable_names == ["x1", "x2"]
        assert X2.generator.numerical_optimizer.n_restarts == 1
        assert np.allclose(
            X2.generator.data[X2.vocs.all_names].to_numpy(),
            X.data[X.vocs.all_names].to_numpy(),
        )
        assert (
            X.generator.model.state_dict().__str__()
            == X2.generator.model.state_dict().__str__()
        )

        X2.step()

    def test_restart(self):
        YAML = """
                dump_file: dump.yml
                generator:
                    name: mobo
                    reference_point: {y1: 1.5, y2: 1.5}
                    numerical_optimizer:
                        name: LBFGS
                        n_restarts: 1

                evaluator:
                    function: xopt.resources.test_functions.tnk.evaluate_TNK

                vocs:
                    variables:
                        x1: [0, 3.14159]
                        x2: [0, 3.14159]
                    objectives: {y1: MINIMIZE, y2: MINIMIZE}
                    constraints: {}
                """
        X = Xopt.from_yaml(YAML)
        X.random_evaluate(3)
        X.step()

        config = yaml.safe_load(open("dump.yml"))

        # test restart
        X2 = Xopt.model_validate(config)

        assert X2.generator.vocs.variable_names == ["x1", "x2"]
        assert X2.generator.numerical_optimizer.n_restarts == 1
        assert np.allclose(
            X2.generator.data[X2.vocs.all_names].to_numpy(),
            X.data[X.vocs.all_names].to_numpy(),
        )

        X2.step()

    @pytest.fixture(scope="module", autouse=True)
    def clean_up(self):
        yield
        files = ["dump.yml", "mobo_model.pt", "dump_inline.yml", "generator_model.pt"]
        for f in files:
            if os.path.exists(f):
                os.remove(f)
