import pytest

from xopt.generators.bayesian.utils import create_constrained_mc_objective
from xopt.resources.testing import TEST_VOCS_BASE
import torch


class TestConstrainedMCObjective:
    def test_basic(self):
        # add a second constraint to the base vocs
        TEST_VOCS_BASE.constraints["c2"] = ["LESS_THAN", 0.5]
        obj = create_constrained_mc_objective(TEST_VOCS_BASE)

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
        ts1 = torch.tensor([-1.0, 10.0, -10.0]).unsqueeze(0).double()
        assert obj(ts1) == torch.tensor(1.0)

        # satisfies one constraint
        ts2 = torch.tensor([-1.0, -10.0, -10.0]).unsqueeze(0).double()
        # satisfies one constraint
        ts3 = torch.tensor([-1.0, 10.0, 10.0]).unsqueeze(0).double()
        # satisfies no constraints
        ts4 = torch.tensor([-1.0, -10.0, 10.0]).unsqueeze(0).double()

        for ele in [ts2, ts3, ts4]:
            assert obj(ele) == torch.tensor(0.0)



