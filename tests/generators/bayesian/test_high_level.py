from copy import deepcopy

import pandas as pd
import torch
import yaml
from xopt import Xopt

from xopt.generators import UpperConfidenceBoundGenerator
from xopt.resources.testing import TEST_VOCS_BASE


class TestHighLevel:
    def test_ucb(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.constraints = {}
        ucb_gen = UpperConfidenceBoundGenerator(test_vocs)
        ucb_gen.options.acq.beta = 0.0
        ucb_gen.options.acq.monte_carlo_samples = 512
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
        xopt: {}
        generator:
            name: mobo
            n_initial: 5
            optim:
                num_restarts: 1
                raw_samples: 2
            acq:
                reference_point: {y1: 1.5, y2: 1.5}
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
        X.step()  # generates random data
        X.step()  # actually evaluates mobo

    def test_mobo(self):
        YAML = """
            xopt: {}
            generator:
                name: mobo
                n_initial: 5
                optim:
                    num_restarts: 2
                    raw_samples: 2
                acq:
                    reference_point: {y1: 1.5, y2: 1.5}

            evaluator:
                function: xopt.resources.test_functions.tnk.evaluate_TNK

            vocs:
                variables:
                    x1: [0, 3.14159]
                    x2: [0, 3.14159]
                objectives: {y1: MINIMIZE, y2: MINIMIZE}
                constraints: {}
        """
        X = Xopt(config=yaml.safe_load(YAML))
        X.step()  # generates random data
        X.step()  # actually evaluates mobo

    def test_restart(self):
        YAML = """
                    xopt: {dump_file: dump.yml}
                    generator:
                        name: mobo
                        n_initial: 5
                        optim:
                            num_restarts: 1
                            raw_samples: 2
                        acq:
                            reference_point: {y1: 1.5, y2: 1.5}
                            proximal_lengthscales: [1.5, 1.5]

                    evaluator:
                        function: xopt.resources.test_functions.tnk.evaluate_TNK

                    vocs:
                        variables:
                            x1: [0, 3.14159]
                            x2: [0, 3.14159]
                        objectives: {y1: MINIMIZE, y2: MINIMIZE}
                        constraints: {}
                """
        X = Xopt(config=yaml.safe_load(YAML))
        X.step()
        X.step()

        # test restart
        X2 = Xopt(config=yaml.safe_load(open("dump.yml")))
        X2.step()

        import os

        os.remove("dump.yml")
