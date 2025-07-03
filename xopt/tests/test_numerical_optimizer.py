import time
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from xopt import Evaluator, Xopt
from xopt.generators.bayesian import BayesianExplorationGenerator
from xopt.numerical_optimizer import (
    GridOptimizer,
    LBFGSOptimizer,
    NumericalOptimizer,
    get_random_ic_generator,
)
from xopt.resources.test_functions.tnk import evaluate_TNK, tnk_vocs
from botorch.optim.parameter_constraints import nonlinear_constraint_is_feasible


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

    def test_lbfgs_optimizer_with_nonlinear_constraints(self):
        # test nonlinear constraints
        def constraint1(X):
            return X[0] + X[1] - 1

        def constraint2(X):
            return X[0] - X[1] + 0.5

        nonlinear_constraints = [constraint1, constraint2]
        optimizer = LBFGSOptimizer(max_time=1.0)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        candidates = optimizer.optimize(
            f,
            bounds,
            n_candidates=3,
            nonlinear_inequality_constraints=nonlinear_constraints,
        )
        assert candidates.shape == torch.Size([3, 2])

        # test case where no feasible points are found
        def infeasible_constraint(X):
            return X[0] + X[1] - 3
        
        nonlinear_constraints.append(infeasible_constraint)
        with pytest.raises(ValueError, match="No valid initial conditions found"):
            optimizer.optimize(
                f,
                bounds,
                n_candidates=3,
                nonlinear_inequality_constraints=nonlinear_constraints,
            )

    # test ic generator
    def test_ic_generator(self):
        # create nonlinear constraints in the botorch style
        # these are not used in the test, but are here to ensure that
        # the generator can handle them
        nonlinear_constraints = [(lambda x: (x[..., 0] - x[..., 1]) ** 2 - 1.0, True)]

        # create a generator with the nonlinear constraints
        random_ic_generator = get_random_ic_generator(nonlinear_constraints)

        # test the generator
        bounds = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        samples = random_ic_generator(None, bounds, 1, 100, 10)
        assert samples.shape == (100, 1, 2)

        for constraint in nonlinear_constraints:
            assert torch.all(
                nonlinear_constraint_is_feasible(constraint[0], constraint[1], samples)
            )

    def test_grid_optimizer(self):
        optimizer = GridOptimizer()
        for ndim in [1, 3]:
            bounds = torch.stack((torch.zeros(ndim), torch.ones(ndim)))
            for ncandidate in [1, 3]:
                candidates = optimizer.optimize(f, bounds, ncandidate)
                assert candidates.shape == torch.Size([ncandidate, ndim])

        # test nonlinear constraints
        def constraint1(X):
            return X[0] + X[1] - 1

        def constraint2(X):
            return X[0] - X[1] + 0.5

        nonlinear_constraints = [constraint1, constraint2]

        optimizer = GridOptimizer(n_grid_points=2)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        candidates = optimizer.optimize(
            f,
            bounds,
            n_candidates=3,
            nonlinear_inequality_constraints=nonlinear_constraints,
        )
        assert candidates.shape == torch.Size([2, 2])

        # test case where no feasible points are found
        def infeasible_constraint(X):
            return X[0] + X[1] - 3

        nonlinear_constraints.append(infeasible_constraint)

        with pytest.raises(ValueError, match="No feasible points found"):
            optimizer.optimize(
                f,
                bounds,
                n_candidates=3,
                nonlinear_inequality_constraints=nonlinear_constraints,
            )

    def test_in_xopt(self):
        vocs = deepcopy(tnk_vocs)

        # can only explore one objective
        vocs.objectives = {}
        vocs.observables = ["y1"]

        generator = BayesianExplorationGenerator(
            vocs=vocs, numerical_optimizer=GridOptimizer(n_grid_points=2)
        )

        evaluator = Evaluator(function=evaluate_TNK)

        X = Xopt(generator=generator, evaluator=evaluator, vocs=vocs)

        X.evaluate_data(
            pd.DataFrame({"x1": [1.0, 0.75, 3.14, 0], "x2": [0.7, 0.95, 0, 3.14]})
        )

        X.step()
        assert np.allclose(
            X.data.iloc[-1][X.vocs.variable_names].to_numpy().astype(float),
            np.zeros(2),
            atol=1e-4,
        )
