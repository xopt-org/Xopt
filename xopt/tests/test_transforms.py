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
