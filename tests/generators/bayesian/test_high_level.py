from copy import deepcopy

import pandas as pd
import torch

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
            assert torch.allclose(acq_val.max(), torch.tensor(9.36).double(), atol=0.1)
