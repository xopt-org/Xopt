from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import PolynomialKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior

from xopt.generators.bayesian.expected_improvement import (
    BayesianOptions,
    ExpectedImprovementGenerator,
)
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.bayesian.options import ModelOptions
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA
from xopt.vocs import VOCS


class TestModelConstructor:
    def test_model_w_nans(self):
        test_data = deepcopy(TEST_VOCS_DATA)
        test_vocs = deepcopy(TEST_VOCS_BASE)
        constructor = StandardModelConstructor(test_vocs, ModelOptions())

        test_data.loc[5, "y1"] = np.nan
        test_data.loc[6, "c1"] = np.nan
        test_data.loc[7, "c1"] = np.nan

        model = constructor.build_model(test_data)

        assert model.train_inputs[0][0].shape == torch.Size([9, 2])
        assert model.train_inputs[1][0].shape == torch.Size([8, 2])

    def test_train_model(self):
        # tests to make sure that models created by StandardModelConstructor class
        # match by-hand botorch SingleTaskGP modules

        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_data = deepcopy(TEST_VOCS_DATA)

        test_pts = torch.tensor(
            pd.DataFrame(TEST_VOCS_BASE.random_inputs(5, False, False)).to_numpy()
        )

        test_covar_modules = []

        # add empty dict to test default covar module
        test_covar_modules += [{}]

        # prepare custom covariance module
        covar_module = PolynomialKernel(power=1, active_dims=[0]) * PolynomialKernel(
            power=1, active_dims=[1]
        )

        scaled_covar_module = ScaleKernel(covar_module)
        covar_module_dict = {"y1": scaled_covar_module}

        test_covar_modules += [covar_module_dict]

        for test_covar in test_covar_modules:
            test_covar1 = deepcopy(test_covar)
            test_covar2 = deepcopy(test_covar)

            # train model with StandardModelConstructor
            model_options = ModelOptions(covar_modules=test_covar1)
            model_constructor = StandardModelConstructor(test_vocs, model_options)
            constructed_model = model_constructor.build_model(test_data[:5]).models[0]

            # build initial model explicitly for comparison
            train_X = torch.cat(
                (
                    torch.tensor(test_data["x1"][:5]).reshape(-1, 1),
                    torch.tensor(test_data["x2"][:5]).reshape(-1, 1),
                ),
                dim=1,
            )
            train_Y = torch.tensor(test_data["y1"][:5]).reshape(-1, 1)
            if test_covar2:
                covar_module = PolynomialKernel(
                    power=1, active_dims=[0]
                ) * PolynomialKernel(power=1, active_dims=[1])
                scaled_covar_module = ScaleKernel(covar_module)
                covar2 = scaled_covar_module
            else:
                covar2 = None

            input_transform = Normalize(
                test_vocs.n_variables, bounds=torch.tensor(test_vocs.bounds)
            )
            benchmark_model = SingleTaskGP(
                train_X,
                train_Y,
                input_transform=input_transform,
                outcome_transform=Standardize(1),
                covar_module=covar2,
                likelihood=GaussianLikelihood(noise_prior=GammaPrior(1.0, 10.0)),
            )

            init_mll = ExactMarginalLogLikelihood(
                benchmark_model.likelihood, benchmark_model
            )
            fit_gpytorch_mll(init_mll)

            assert torch.allclose(
                benchmark_model.train_inputs[0], constructed_model.train_inputs[0]
            )
            assert torch.allclose(
                benchmark_model.train_targets, constructed_model.train_targets
            )

            if test_covar2:
                assert torch.allclose(
                    benchmark_model.covar_module.base_kernel.kernels[0].offset,
                    constructed_model.covar_module.base_kernel.kernels[0].offset,
                )

            with torch.no_grad():
                constructed_prediction = constructed_model.posterior(test_pts).mean
                benchmark_prediction = benchmark_model.posterior(test_pts).mean

            assert torch.allclose(
                constructed_prediction, benchmark_prediction, rtol=1e-3
            )

    def test_train_from_scratch(self):
        # test to verify that GP modules are trained from scratch everytime
        # avoids training pitfalls due to local minima in likelihoods due to smaller
        # data sets -- relevant for low order kernels
        var_names = ["x0", "x1"]

        def centroid_position_at_screen(x):
            r0 = 0.0
            cpas = (r0 + x[:, 0]) + (r0 + x[:, 0]) * x[:, 1]

            #     return cpas * (1. + .1*torch.randn_like(cpas))
            return cpas

        def test_func(input_dict):
            x0 = torch.tensor(input_dict["x0"]).reshape(-1, 1)
            x1 = torch.tensor(input_dict["x1"]).reshape(-1, 1)
            x = torch.cat((x0, x1), dim=1)
            return {"y": centroid_position_at_screen(x).squeeze().cpu().numpy()}

        variables = {var_name: [-2, 2] for var_name in var_names}

        # construct vocs
        vocs = VOCS(variables=variables, objectives={"y": "MINIMIZE"})

        # prepare custom covariance module
        covar_module = PolynomialKernel(power=1, active_dims=[0]) * PolynomialKernel(
            power=1, active_dims=[1]
        )
        scaled_covar_module = ScaleKernel(covar_module)

        # prepare options for Xopt generator
        covar_module_dict = {"y": scaled_covar_module}
        model_options = ModelOptions(covar_modules=covar_module_dict)

        # construct BAX generator
        generator = ExpectedImprovementGenerator(
            vocs, BayesianOptions(model=deepcopy(model_options))
        )

        # define test points
        # test equivalence
        bounds = vocs.bounds
        n = 10
        x = torch.linspace(*bounds.T[0], n)
        y = torch.linspace(*bounds.T[1], n)
        xx, yy = torch.meshgrid(x, y)
        test_pts = torch.hstack([ele.reshape(-1, 1) for ele in (xx, yy)]).double()

        # create input points that will produce a broad local extrema that adding
        # points will not escape IF TRAINING THE HYPERPARAMETERS IS NOT DONE FROM
        # SCRATCH
        inputs = {"x0": [-1.5, -1.2], "x1": [0.0, 0.0]}
        outputs = test_func(inputs)
        data = pd.DataFrame(inputs).join(pd.DataFrame(outputs))

        # this training should find a local extrema in hyperparameter space
        generator.add_data(data)
        generated_model = generator.train_model()

        # get old prediction
        with torch.no_grad():
            old_prediction = generated_model.posterior(test_pts).mean[..., 0]

        # adding these points should change the prediction
        inputs = {"x0": [1.2, 0.0], "x1": [0.0, 1.0]}
        outputs = test_func(inputs)
        data = pd.DataFrame(inputs).join(pd.DataFrame(outputs))

        generator.add_data(data)
        generated_model = generator.train_model()

        # construct generator with all points
        generator = ExpectedImprovementGenerator(
            vocs, BayesianOptions(model=deepcopy(model_options))
        )

        # create  input points
        total_inputs = {"x0": [-1.5, -1.2, 1.2, 0.0], "x1": [0.0, 0.0, 0.0, 1.0]}
        total_outputs = test_func(total_inputs)
        total_data = pd.DataFrame(total_inputs).join(pd.DataFrame(total_outputs))

        generator.add_data(total_data)
        benchmark_model = generator.train_model()

        # make sure models have exactly the same data points
        assert torch.allclose(
            benchmark_model.models[0].train_inputs[0],
            generated_model.models[0].train_inputs[0],
        )
        assert torch.allclose(
            benchmark_model.models[0].train_targets,
            generated_model.models[0].train_targets,
        )

        with torch.no_grad():
            generated_pred = generated_model.posterior(test_pts).mean[..., 0]
            benchmark_pred = benchmark_model.posterior(test_pts).mean[..., 0]

            assert torch.allclose(generated_pred, benchmark_pred, rtol=1e-3)
            assert ~torch.allclose(generated_pred, old_prediction, rtol=1e-3)
