import pytest
import time
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from xopt import Evaluator, Xopt
from xopt.generators.bayesian import BayesianExplorationGenerator
from xopt.numerical_optimizer import GridOptimizer, LBFGSOptimizer, NumericalOptimizer
from xopt.resources.test_functions.tnk import evaluate_TNK, tnk_vocs


def f(X):
    # X: A `batch_shape x q x d` Tensor of t-batches with `q` `d`-dim design points each.
    # _sample_forward would produce `sample_shape x batch_shape x q` (sample shape set by MC sampler)
    # q_reduction converts it into `sample_shape x batch_shape`
    # sample_reduction converts it into `batch_shape`-dim Tensor of acquisition values
    # so, final fake tensor needs to have ndim=1
    result = torch.amax(X, dim=(1, 2))
    return result


class TestNumericalOptimizers:
    @patch.multiple(NumericalOptimizer, __abstractmethods__=set())
    def test_base(self):
        NumericalOptimizer()

    def test_lbfgs_optimizer(self):
        optimizer = LBFGSOptimizer()
        for ndim in [1, 3]:
            bounds = torch.stack((torch.zeros(ndim), torch.ones(ndim)))
            for ncandidate in [1, 3]:
                candidates = optimizer.optimize(
                    function=f, bounds=bounds, n_candidates=ncandidate
                )
                assert candidates.shape == torch.Size([ncandidate, ndim])

        # test max time
        max_time_optimizer = LBFGSOptimizer(max_time=1.0)
        ndim = 1
        bounds = torch.stack((torch.zeros(ndim), torch.ones(ndim)))
        for ncandidate in [1, 3]:
            start_time = time.time()
            candidates = max_time_optimizer.optimize(f, bounds, ncandidate)
            assert time.time() - start_time < 1.0
            assert candidates.shape == torch.Size([ncandidate, ndim])

    def test_lbfgsoptimizer_max_time_none(self):
        optimizer = LBFGSOptimizer(max_time=None)
        bounds = torch.tensor([[0.0], [1.0]])
        with patch("xopt.numerical_optimizer.optimize_acqf") as mock_opt:
            mock_opt.return_value = (torch.zeros(1, 1), None)
            candidates = optimizer.optimize(f, bounds, n_candidates=1)
            assert candidates.shape == (1, 1)
            # Ensure timeout_sec is None
            assert mock_opt.call_args[1]["timeout_sec"] is None

    def test_lbfgsoptimizer_bounds_shape_error(self):
        optimizer = LBFGSOptimizer()
        # bounds with wrong shape: should be [2, ndim], here [3, 1]
        bad_bounds = torch.zeros(3, 1)
        with pytest.raises(ValueError, match="bounds must have the shape"):
            optimizer.optimize(f, bad_bounds, n_candidates=1)

    def test_grid_optimizer(self):
        optimizer = GridOptimizer()
        for ndim in [1, 3]:
            bounds = torch.stack((torch.zeros(ndim), torch.ones(ndim)))
            for ncandidate in [1, 3]:
                candidates = optimizer.optimize(f, bounds, ncandidate)
                assert candidates.shape == torch.Size([ncandidate, ndim])

    def test_gridoptimizer_bounds_shape_error(self):
        optimizer = GridOptimizer()
        # bounds with wrong shape: should be [2, ndim], here [3, 1]
        bad_bounds = torch.zeros(3, 1)
        with pytest.raises(ValueError, match="bounds must have the shape"):
            optimizer.optimize(f, bad_bounds, n_candidates=1)

    def test_in_xopt(self):
        vocs = deepcopy(tnk_vocs)

        # can only explore one objective
        vocs.objectives = {"y1": "EXPLORE"}

        generator = BayesianExplorationGenerator(
            vocs=vocs, numerical_optimizer=GridOptimizer(n_grid_points=2)
        )

        evaluator = Evaluator(function=evaluate_TNK)

        X = Xopt(generator=generator, evaluator=evaluator)

        X.evaluate_data(
            pd.DataFrame({"x1": [1.0, 0.75, 3.14, 0], "x2": [0.7, 0.95, 0, 3.14]})
        )

        X.step()
        assert np.allclose(
            X.data.iloc[-1][X.vocs.variable_names].to_numpy().astype(float),
            np.zeros(2),
            atol=1e-4,
        )
