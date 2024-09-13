from copy import deepcopy

import torch

from xopt.generators.bayesian.objectives import create_mobo_objective
from xopt.resources.testing import TEST_VOCS_BASE


class TestUtils:
    def test_mobo_objective(self):
        test_vocs_copy = deepcopy(TEST_VOCS_BASE)
        test_vocs_copy.objectives["y2"] = "MAXIMIZE"
        obj = create_mobo_objective(test_vocs_copy)

        # test large sample shape
        test_samples = torch.randn(3, 4, 5, 3).double()
        output = obj(test_samples)
        assert output.shape == torch.Size([3, 4, 5, 2])

        # test to make sure values are correct - minimize axis should be negated
        test_samples = torch.rand(5, 4, 3)
        output = obj(test_samples)
        assert torch.allclose(output[..., 1], test_samples[..., 1])
        assert torch.allclose(output[..., 0], -test_samples[..., 0])
