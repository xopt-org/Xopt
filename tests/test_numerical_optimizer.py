from unittest.mock import patch

import torch

from xopt.numerical_optimizer import NumericalOptimizer, LBFGSOptimizer, GridOptimizer


def f(X):
    return torch.sum(X**2, dim=-1)


class TestNumericalOptimizers:
    @patch.multiple(NumericalOptimizer, __abstractmethods__=set())
    def test_base(self):
        NumericalOptimizer()

    def test_lbfgs_optimizer(self):
        optimizer = LBFGSOptimizer()
        for ndim in [1, 3]:
            bounds = torch.stack((torch.zeros(ndim), torch.ones(ndim)))
            for ncandidate in [1, 3]:
                candidates = optimizer.optimize(f, bounds, ncandidate)
                assert candidates.shape == torch.Size([ncandidate, ndim])

    def test_grid_optimizer(self):
        optimizer = GridOptimizer()
        for ndim in [1, 3]:
            bounds = torch.stack((torch.zeros(ndim), torch.ones(ndim)))
            for ncandidate in [1, 3]:
                candidates = optimizer.optimize(f, bounds, ncandidate)
                assert candidates.shape == torch.Size([ncandidate, ndim])

