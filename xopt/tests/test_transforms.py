import pytest

from ..bayesian.outcome_transforms import NanEnabledStandardize
from ..bayesian.input_transforms import CostAwareNormalize
import torch
import numpy as np


class TestClassStandardize:
    def test_nan_standardize(self):
        data = np.random.rand(5, 3)
        data[0][1] = np.nan
        data = torch.tensor(data)

        s = NanEnabledStandardize(3)
        s.forward(data)


class TestClassNormalize:
    def test_cost_aware_normalize(self):
        data = torch.rand(5, 3)
        trans = CostAwareNormalize(3)
        trans_data = trans._transform(data)
        assert torch.all(trans_data[..., -1] == data[..., -1])
        assert torch.all(data[..., -1] == trans._untransform(trans_data)[..., -1])
