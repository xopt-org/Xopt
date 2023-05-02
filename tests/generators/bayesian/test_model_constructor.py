import torch
import numpy as np
from copy import deepcopy

from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.resources.testing import TEST_VOCS_DATA, TEST_VOCS_BASE
from xopt.generators.bayesian.options import ModelOptions


class TestModelConstructor:
    def test_model_w_nans(self):
        test_data = deepcopy(TEST_VOCS_DATA)
        test_vocs = deepcopy(TEST_VOCS_BASE)
        constructor = StandardModelConstructor(test_vocs, ModelOptions())

        test_data.loc[5, "y1"] = np.nan
        test_data.loc[6, "c1"] = np.nan
        test_data.loc[7, "c1"] = np.nan

        model = constructor.build_model(test_data)

        assert model.train_inputs[0][0].shape == torch.Size([9, 2])
        assert model.train_inputs[1][0].shape == torch.Size([8, 2])
