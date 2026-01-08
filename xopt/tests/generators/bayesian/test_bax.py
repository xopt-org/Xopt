import pickle
from copy import deepcopy
from unittest.mock import patch

import pytest

import torch
from botorch.models import SingleTaskGP
from botorch.models.model import ModelList
from botorch.models.transforms import Normalize, Standardize


from xopt.base import Xopt
from xopt.errors import VOCSError
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.bax.algorithms import (
    Algorithm,
    GridOptimize,
    GridScanAlgorithm,
    CurvatureGridOptimize,
)
from xopt.generators.bayesian.bax_generator import BaxGenerator
from xopt.generators.bayesian.bax.visualize import visualize_virtual_objective
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA, xtest_callable


class TestBaxGenerator:
    @patch.multiple(Algorithm, __abstractmethods__=set())
    def test_init(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives = {}
        test_vocs.observables = ["y1"]
        alg = Algorithm()
        bax_gen = BaxGenerator(
            vocs=test_vocs,
            algorithm=alg,
        )
        bax_gen.model_dump()

    @patch.multiple(Algorithm, __abstractmethods__=set())
    def test_base_algorithm(self):
        # test abstract algorithm
        Algorithm()

    @patch.multiple(GridScanAlgorithm, __abstractmethods__=set())
    def test_grid_scan_algorithm(self):
        # test abstract grid scan algorithm
        for ndim in [1, 3]:
            bounds = torch.stack([torch.zeros(ndim), torch.ones(ndim)])
            alg = GridScanAlgorithm()
            mesh = alg.create_mesh(bounds=bounds)

            # benchmark
            n_mesh = alg.n_mesh_points
            linspace_list = [
                torch.linspace(bounds.T[i][0], bounds.T[i][1], n_mesh)
                for i in range(ndim)
            ]
            xx = torch.meshgrid(*linspace_list, indexing="ij")
            benchmark_mesh = torch.stack(xx).flatten(start_dim=1).T
            assert torch.allclose(mesh, benchmark_mesh)
            assert mesh.shape == torch.Size([10**ndim, ndim])

    @patch.multiple(GridScanAlgorithm, __abstractmethods__=set())
    def test_create_mesh_invalid_bounds(self):
        alg = GridScanAlgorithm()
        # bounds with wrong shape (should be [2, ndim])
        invalid_bounds = torch.zeros(3, 2)  # shape [3, 2] instead of [2, ndim]
        with pytest.raises(ValueError, match=r"bounds must have the shape \[2, ndim\]"):
            alg.create_mesh(invalid_bounds)

    def test_algorithm_get_execution_paths_not_implemented(self):
        class DummyAlgorithm(Algorithm):
            def evaluate_virtual_objective(
                self, model, x, bounds, n_samples, tkwargs=None
            ):
                return torch.zeros(1)

            def get_execution_paths(self, model, bounds):
                return super().get_execution_paths(model, bounds)

        alg = DummyAlgorithm()
        alg.get_execution_paths(None, None)

    def test_grid_minimize(self):
        # test grid scan minimize
        for ndim in [1, 3]:
            bounds = torch.stack([torch.zeros(ndim), torch.ones(ndim)])
            alg = GridOptimize()

            # create a ModelList with a single output
            train_X = torch.rand(10, ndim, dtype=torch.float64)
            train_Y = torch.rand(10, 1, dtype=torch.float64)

            model = ModelList(
                SingleTaskGP(
                    train_X,
                    train_Y,
                    input_transform=Normalize(ndim),
                    outcome_transform=Standardize(1),
                )
            )

            x_exe, y_exe, results = alg.get_execution_paths(model, bounds)

            # execution paths should be able to be transformed and used in
            # `condition_on_observations` method
            assert x_exe.shape == torch.Size([alg.n_samples, 1, ndim])
            assert y_exe.shape == torch.Size([alg.n_samples, 1, 1])

            # need to call each sub-model on some data before conditioning
            [m(x_exe) for m in model.models]

            x_exe_t = [
                model.models[i].input_transform(x_exe) for i in range(len(model.models))
            ]
            y_exe_t = [
                model.models[i].outcome_transform(
                    torch.index_select(y_exe, dim=-1, index=torch.tensor([i]))
                )[0]
                for i in range(len(model.models))
            ]
            fantasy_models = [
                m.condition_on_observations(x, y)
                for m, x, y in zip(model.models, x_exe_t, y_exe_t)
            ]

            # validate fantasy models
            for fantasy_model in fantasy_models:
                assert fantasy_model.batch_shape == torch.Size([alg.n_samples])
                assert fantasy_model.train_inputs[0].shape == torch.Size(
                    [alg.n_samples, 11, ndim]
                )
                assert fantasy_model.train_targets.shape == torch.Size(
                    [alg.n_samples, 11]
                )

    def test_grid_optimize_maximize(self):
        for ndim in [1, 3]:
            bounds = torch.stack([torch.zeros(ndim), torch.ones(ndim)])
            train_X = torch.rand(10, ndim, dtype=torch.float64)
            train_Y = torch.rand(10, 1, dtype=torch.float64)
            model = SingleTaskGP(
                train_X,
                train_Y,
                input_transform=Normalize(ndim),
                outcome_transform=Standardize(1),
            )
            alg = GridOptimize(minimize=False)
            x_exe, y_exe, results = alg.get_execution_paths(model, bounds)
            # Should select the maximum value from posterior_samples
            posterior_samples = results["posterior_samples"]
            y_max, idx_max = torch.max(posterior_samples, dim=-2)
            assert torch.allclose(y_exe.squeeze(-2), y_max)
            assert x_exe.shape[0] == alg.n_samples

    def test_generate(self):
        alg = GridOptimize()

        # test w/o constraints
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives = {}
        test_vocs.observables = ["y1"]
        test_vocs.constraints = {}
        gen = BaxGenerator(
            vocs=test_vocs,
            algorithm=alg,
            algorithm_results_file="test",
        )
        gen.numerical_optimizer.n_restarts = 1
        gen.data = TEST_VOCS_DATA

        candidate = gen.generate(1)
        assert len(candidate) == 1

        # test w/ constraints
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives = {}
        test_vocs.observables = ["y1"]
        gen = BaxGenerator(
            vocs=test_vocs,
            algorithm=alg,
        )
        gen.numerical_optimizer.n_restarts = 1
        gen.data = TEST_VOCS_DATA

        candidate = gen.generate(1)
        assert len(candidate) == 1

    def test_cuda(self):
        alg = GridOptimize()
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives = {}
        test_vocs.observables = ["y1"]
        gen = BaxGenerator(
            vocs=test_vocs,
            algorithm=alg,
        )

        if torch.cuda.is_available():
            gen.numerical_optimizer.n_restarts = 1
            gen.data = TEST_VOCS_DATA

            candidate = gen.generate(1)
            assert len(candidate) == 1

    def test_in_xopt(self):
        evaluator = Evaluator(function=xtest_callable)
        alg = GridOptimize()

        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives = {}
        test_vocs.observables = ["y1"]
        gen = BaxGenerator(
            vocs=test_vocs,
            algorithm=alg,
        )
        gen.numerical_optimizer.n_restarts = 1

        xopt = Xopt(generator=gen, evaluator=evaluator, vocs=test_vocs)

        # initialize with single initial candidate
        xopt.random_evaluate(3)
        xopt.step()

    def test_file_saving(self):
        evaluator = Evaluator(function=xtest_callable)
        alg = GridOptimize()

        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives = {}
        test_vocs.observables = ["y1"]
        gen = BaxGenerator(vocs=test_vocs, algorithm=alg, algorithm_results_file="test")
        gen.numerical_optimizer.n_restarts = 1

        xopt = Xopt(generator=gen, evaluator=evaluator, vocs=test_vocs)

        # initialize with single initial candidate
        xopt.random_evaluate(3)
        xopt.step()
        xopt.step()

        # test loading saved info
        with open("test_2.pkl", "rb") as handle:
            result = pickle.load(handle)

        assert isinstance(result["test_points"], torch.Tensor)
        assert isinstance(result["posterior_samples"], torch.Tensor)
        assert isinstance(result["execution_paths"], torch.Tensor)

        import os

        os.remove("test_1.pkl")
        os.remove("test_2.pkl")

    def test_vocs_validation(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        alg = GridOptimize()

        with pytest.raises(VOCSError):
            BaxGenerator(vocs=test_vocs, algorithm=alg)

    def test_curvature_grid_optimize_virtual_objective(self):
        ndim = 2
        bounds = torch.stack([torch.zeros(ndim), torch.ones(ndim)])
        train_X = torch.rand(10, ndim, dtype=torch.float64)
        train_Y = torch.rand(10, 1, dtype=torch.float64)
        model = SingleTaskGP(
            train_X,
            train_Y,
            input_transform=Normalize(ndim),
            outcome_transform=Standardize(1),
        )
        alg = CurvatureGridOptimize(n_samples=3, use_mean=False)
        mesh = alg.create_mesh(bounds)
        result = alg.evaluate_virtual_objective(model, mesh, bounds, n_samples=3)
        # Should have shape [n_samples, mesh_points, 1]
        assert result.shape[0] == 3
        assert result.shape[1] == mesh.shape[0]
        # Edge values should be zero
        assert torch.all(result[:, 0] == 0)
        assert torch.all(result[:, -1] == 0)

    def test_curvature_grid_optimize_use_mean(self):
        ndim = 1
        bounds = torch.stack([torch.zeros(ndim), torch.ones(ndim)])
        train_X = torch.rand(10, ndim, dtype=torch.float64)
        train_Y = torch.rand(10, 1, dtype=torch.float64)
        model = SingleTaskGP(
            train_X,
            train_Y,
            input_transform=Normalize(ndim),
            outcome_transform=Standardize(1),
        )
        alg = CurvatureGridOptimize(n_samples=2, use_mean=True)
        mesh = alg.create_mesh(bounds)
        result = alg.evaluate_virtual_objective(model, mesh, bounds, n_samples=2)
        # Should have shape [1, mesh_points, 1] since use_mean=True
        assert result.shape[0] == 1
        assert result.shape[1] == mesh.shape[0]
        # Edge values should be zero
        assert torch.all(result[:, 0] == 0)
        assert torch.all(result[:, -1] == 0)

    def test_visualization(self):
        evaluator = Evaluator(function=xtest_callable)
        alg = GridOptimize()

        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives = {}
        test_vocs.observables = ["y1"]
        gen = BaxGenerator(
            vocs=test_vocs,
            algorithm=alg,
            n_monte_carlo_samples=10
        )
        gen.numerical_optimizer.n_restarts = 1

        xopt = Xopt(generator=gen, evaluator=evaluator, vocs=test_vocs)

        # initialize with single initial candidate
        xopt.random_evaluate(3)
        xopt.step()

        visualize_virtual_objective(generator=xopt.generator)

        with pytest.raises(ValueError):
            visualize_virtual_objective(
                generator=xopt.generator, variable_names=["x1", "x2", "x3"]
            )

        with pytest.raises(ValueError):
            visualize_virtual_objective(
                generator=xopt.generator, variable_names=["invalid_name"]
            )

        with pytest.raises(ValueError):
            visualize_virtual_objective(
                generator=xopt.generator, reference_point={"invalid_name": 0.5}
            )

        visualize_virtual_objective(
            generator=xopt.generator, variable_names=["x1"], n_samples=5
        )

        xopt.generator.model = None
        with pytest.raises(ValueError):
            visualize_virtual_objective(generator=xopt.generator)
