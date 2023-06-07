from unittest.mock import patch

import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize

from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.bax.algorithms import (
    Algorithm,
    GridMinimize,
    GridScanAlgorithm,
)
from xopt.generators.bayesian.bax_generator import BaxGenerator
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA, xtest_callable


class TestBaxGenerator:
    @patch.multiple(Algorithm, __abstractmethods__=set())
    def test_init(self):
        alg = Algorithm()
        bax_gen = BaxGenerator(
            vocs=TEST_VOCS_BASE,
            algorithm=alg,
        )
        bax_gen.dict()

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

    def test_grid_minimize(self):
        # test grid scan minimize
        for ndim in [1, 3]:
            bounds = torch.stack([torch.zeros(ndim), torch.ones(ndim)])
            alg = GridMinimize()

            # create a model
            train_X = torch.rand(10, ndim)
            train_Y = torch.rand(10, 1)

            model = SingleTaskGP(
                train_X,
                train_Y,
                input_transform=Normalize(ndim),
                outcome_transform=Standardize(1),
            )

            x_exe, y_exe, results = alg.get_execution_paths(model, bounds)

            # execution paths should be able to be transformed and used in
            # `condition_on_observations` method
            assert x_exe.shape == torch.Size([alg.n_samples, 1, ndim])
            assert y_exe.shape == torch.Size([alg.n_samples, 1, 1])

            # need to call model on some data before conditioning
            model(x_exe)

            xs_exe_transformed = model.input_transform(x_exe)
            ys_exe_transformed = model.outcome_transform(y_exe)[0]
            fantasy_models = model.condition_on_observations(
                xs_exe_transformed, ys_exe_transformed
            )

            # validate fantasy model
            assert fantasy_models.batch_shape == torch.Size([alg.n_samples])
            assert fantasy_models.train_inputs[0].shape == torch.Size(
                [alg.n_samples, 11, ndim]
            )
            assert fantasy_models.train_targets.shape == torch.Size([alg.n_samples, 11])

    def test_generate(self):
        alg = GridMinimize()

        gen = BaxGenerator(
            vocs=TEST_VOCS_BASE,
            algorithm=alg,
        )
        gen.optimization_options.raw_samples = 1
        gen.optimization_options.num_restarts = 1
        gen.data = TEST_VOCS_DATA

        candidate = gen.generate(1)
        assert len(candidate) == 1

    def test_cuda(self):
        alg = GridMinimize()
        gen = BaxGenerator(
            vocs=TEST_VOCS_BASE,
            algorithm=alg,
        )

        if torch.cuda.is_available():
            gen.optimization_options.raw_samples = 1
            gen.optimization_options.num_restarts = 1
            gen.data = TEST_VOCS_DATA

            candidate = gen.generate(1)
            assert len(candidate) == 1

    def test_in_xopt(self):
        evaluator = Evaluator(function=xtest_callable)
        alg = GridMinimize()

        gen = BaxGenerator(
            vocs=TEST_VOCS_BASE,
            algorithm=alg,
        )
        gen.optimization_options.raw_samples = 1
        gen.optimization_options.num_restarts = 1

        xopt = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)

        # initialize with single initial candidate
        xopt.random_evaluate(3)
        xopt.step()
