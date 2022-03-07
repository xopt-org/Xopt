from xopt.generators.bayesian.utils import create_constrained_mc_objective
from xopt.resources.testing import TEST_VOCS_BASE
import torch


class TestConstrainedMCObjective:
    def test_basic(self):
        obj = create_constrained_mc_objective(TEST_VOCS_BASE)

        test_samples = torch.randn(3, 4, 5, 2).double()*10.0 + 0.5

        output = obj(test_samples)
        print(output)
        #assert torch.allclose(correct_output, output)
