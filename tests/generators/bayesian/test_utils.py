import torch

from xopt.generators.bayesian.utils import create_constrained_mc_objective
from xopt.resources.testing import TEST_VOCS_BASE


class TestBayesianUtils:
    def test_create_constrained_mc_objective(self):
        TEST_VOCS_BASE.constraints.update({"c2": ["LESS_THAN", 0.75]})

        obj = create_constrained_mc_objective(TEST_VOCS_BASE)
        samples = torch.rand(512, 10, 1, 3, dtype=torch.double)

        out = obj(samples)
        assert out.shape == torch.Size([512, 10, 1])
