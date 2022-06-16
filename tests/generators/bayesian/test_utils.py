from copy import deepcopy

import pytest
import torch

from xopt.generators.bayesian.objectives import (
    create_constrained_mc_objective,
    create_mobo_objective,
)
from xopt.resources.testing import TEST_VOCS_BASE


class TestUtils:
    def test_constrained_mc_objective(self):
        # add a second constraint to the base vocs
        test_vocs_copy = deepcopy(TEST_VOCS_BASE)
        test_vocs_copy.constraints["c2"] = ["LESS_THAN", 0.5]
        obj = create_constrained_mc_objective(test_vocs_copy)

        # test large sample shape
        test_samples = torch.randn(3, 4, 5, 3).double()
        output = obj(test_samples)
        assert output.shape == torch.Size([3, 4, 5])

        # test bad sample shape
        with pytest.raises(RuntimeError):
            test_samples = torch.randn(3, 4, 5, 2).double()
            output = obj(test_samples)

        # test explicit points (posterior samples)
        # satisfies both constraints
        ts1 = torch.tensor([-1.0, -10.0, -10.0]).unsqueeze(0).double()
        assert obj(ts1) == torch.tensor(1.0)

        # satisfies one constraint
        ts2 = torch.tensor([-1.0, -10.0, 10.0]).unsqueeze(0).double()
        # satisfies one constraint
        ts3 = torch.tensor([-1.0, 10.0, -10.0]).unsqueeze(0).double()
        # satisfies no constraints
        ts4 = torch.tensor([-1.0, 10.0, 10.0]).unsqueeze(0).double()

        for ele in [ts2, ts3, ts4]:
            assert obj(ele) == torch.tensor(-100.0)

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
