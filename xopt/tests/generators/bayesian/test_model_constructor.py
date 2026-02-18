import json
import os
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim.fit import fit_gpytorch_mll_torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import PeriodicKernel, PolynomialKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import GammaPrior
from pydantic import ValidationError

from xopt.generators.bayesian.custom_botorch.heteroskedastic import (
    XoptHeteroskedasticSingleTaskGP,
)
from xopt.generators.bayesian.expected_improvement import ExpectedImprovementGenerator
from xopt.generators.bayesian.models.standard import (
    BatchedModelConstructor,
    LBFGSNumericalOptimizerConfig,
    StandardModelConstructor,
)
from xopt.generators.bayesian.utils import get_training_data_batched
from xopt.resources.testing import (
    TEST_VOCS_BASE_3D,
    TEST_VOCS_DATA_3D,
    verify_state_device,
)
from xopt.vocs import VOCS

cuda_combinations = [False] if not torch.cuda.is_available() else [False, True]
device_map = {False: torch.device("cpu"), True: torch.device("cuda:0")}

TEST_VOCS = TEST_VOCS_BASE_3D
TEST_DATA = TEST_VOCS_DATA_3D


class TestModelConstructor:
    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_standard(self, use_cuda):
        test_data = deepcopy(TEST_DATA)
        test_vocs = deepcopy(TEST_VOCS)
        constructor = StandardModelConstructor()

        model = constructor.build_model_from_vocs(
            test_vocs, test_data, device=device_map[use_cuda]
        )
        verify_state_device(model, device=device_map[use_cuda])

    def test_standard_with_numerical(self):
        test_data = deepcopy(TEST_DATA)
        test_vocs = deepcopy(TEST_VOCS)
        constructor = StandardModelConstructor(
            train_config=LBFGSNumericalOptimizerConfig(gtol=1e-3)
        )
        constructor.build_model_from_vocs(test_vocs, test_data)

    def test_standard_timeout(self):
        test_data = deepcopy(TEST_DATA)
        test_vocs = deepcopy(TEST_VOCS)
        data = pd.concat([test_data] * 20, ignore_index=True)
        data.loc[:, test_vocs.objective_names] += 0.1 * np.random.randn(
            len(test_data) * 20, test_vocs.n_objectives
        )
        constructor = StandardModelConstructor(
            train_config=LBFGSNumericalOptimizerConfig(timeout=0.01),
        )
        t1 = time.perf_counter()
        constructor.build_model_from_vocs(test_vocs, data)
        t2 = time.perf_counter()
        delta = t2 - t1
        assert delta < 0.3

    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_standard_adam(self, use_cuda):
        test_data = deepcopy(TEST_DATA)
        test_vocs = deepcopy(TEST_VOCS)
        constructor = StandardModelConstructor(train_method="adam")
        # monkeypatch to check if adam was called
        import xopt.generators.bayesian.models.standard as st

        called = False

        def torch_monkeypatch(*args, **kwargs):
            nonlocal called
            called = True
            return fit_gpytorch_mll_torch(*args, **kwargs)

        st.fit_gpytorch_mll_torch = torch_monkeypatch
        model = constructor.build_model_from_vocs(
            test_vocs, test_data, device=device_map[use_cuda]
        )
        verify_state_device(model, device=device_map[use_cuda])
        assert called

    def test_transform_inputs(self):
        test_data = deepcopy(TEST_DATA)
        test_vocs = deepcopy(TEST_VOCS)

        # test case where no inputs are transformed
        constructor = StandardModelConstructor(transform_inputs=False)
        model = constructor.build_model(
            test_vocs.variable_names, test_vocs.output_names, test_data
        )
        assert not hasattr(model.models[0], "input_transform")

        # test case where only one input is transformed
        constructor = StandardModelConstructor(transform_inputs={"c1": False})
        model = constructor.build_model(
            test_vocs.variable_names, test_vocs.output_names, test_data
        )
        assert hasattr(model.models[0], "input_transform")
        assert not hasattr(model.models[1], "input_transform")

    def test_duplicate_keys(self):
        test_data = deepcopy(TEST_DATA)
        test_vocs = deepcopy(TEST_VOCS)
        test_vocs.observables = ["y1"]

        constructor = StandardModelConstructor()

        constructor.build_model(
            test_vocs.variable_names, test_vocs.output_names, test_data
        )

        model = constructor.build_model_from_vocs(test_vocs, test_data)
        assert model.num_outputs == 2

    def test_custom_model(self):
        test_data = deepcopy(TEST_DATA)
        test_vocs = deepcopy(TEST_VOCS)

        custom_covar = {"y1": ScaleKernel(PeriodicKernel())}

        with pytest.raises(ValidationError):
            StandardModelConstructor(
                vocs=test_vocs, covar_modules=deepcopy(custom_covar)["y1"]
            )

        # test custom covar module
        constructor = StandardModelConstructor(covar_modules=deepcopy(custom_covar))
        model = constructor.build_model(
            test_vocs.variable_names, test_vocs.output_names, test_data
        )
        assert isinstance(model.models[0].covar_module.base_kernel, PeriodicKernel)

        # test prior mean
        class ConstraintPrior(torch.nn.Module):
            def forward(self, X):
                return X[:, 0] ** 2

        mean_modules = {"c1": ConstraintPrior()}
        constructor = StandardModelConstructor(mean_modules=mean_modules)
        model = constructor.build_model_from_vocs(test_vocs, test_data)
        assert isinstance(model.models[1].mean_module.model, ConstraintPrior)

    def test_model_w_nans(self):
        test_data = deepcopy(TEST_DATA)
        test_vocs = deepcopy(TEST_VOCS)
        constructor = StandardModelConstructor()

        # add nans to ouputs
        test_data.loc[5, "y1"] = np.nan
        test_data.loc[6, "c1"] = np.nan
        test_data.loc[7, "c1"] = np.nan

        model = constructor.build_model_from_vocs(test_vocs, test_data)

        assert model.train_inputs[0][0].shape == torch.Size([9, test_vocs.n_variables])
        assert model.train_inputs[1][0].shape == torch.Size([8, test_vocs.n_variables])

        # add nans to inputs
        test_data2 = deepcopy(TEST_DATA)
        test_data2.loc[5, "x1"] = np.nan

        model2 = constructor.build_model_from_vocs(test_vocs, test_data2)
        assert model2.train_inputs[0][0].shape == torch.Size([9, test_vocs.n_variables])

        # add nans to both
        test_data3 = deepcopy(TEST_DATA)
        test_data3.loc[5, "x1"] = np.nan
        test_data3.loc[7, "c1"] = np.nan

        model3 = constructor.build_model_from_vocs(test_vocs, test_data3)
        assert model3.train_inputs[0][0].shape == torch.Size([9, test_vocs.n_variables])
        assert model3.train_inputs[1][0].shape == torch.Size([8, test_vocs.n_variables])

    def test_model_w_same_data(self):
        test_data = deepcopy(TEST_DATA)
        test_vocs = deepcopy(TEST_VOCS)
        test_vocs.variables["x1"] = [5.0, 6.0]
        constructor = StandardModelConstructor()

        # set all of the elements of a given input variable to the same value
        test_data["x1"] = 5.0

        constructor.build_model_from_vocs(test_vocs, test_data)

    def test_serialization(self):
        # test custom covar module
        custom_covar = {"y1": ScaleKernel(PeriodicKernel())}
        constructor = StandardModelConstructor(covar_modules=custom_covar)
        constructor.to_json(serialize_torch=True)
        os.remove("covar_modules_y1.pt")

    def test_model_saving(self):
        my_vocs = VOCS(
            variables={"x": [0, 1]},
            objectives={"y": "MAXIMIZE"},
            constraints={"c": ["LESS_THAN", 0]},
        )

        # specify a periodic kernel for each output (objectives and constraints)
        covar_modules = {"y": ScaleKernel(PeriodicKernel())}

        gp_constructor = StandardModelConstructor(covar_modules=covar_modules)
        generator = ExpectedImprovementGenerator(
            vocs=my_vocs, gp_constructor=gp_constructor
        )

        # define training data to pass to the generator
        train_x = torch.tensor((0.2, 0.5, 0.6))
        train_y = 5.0 * torch.cos(2 * 3.14 * train_x + 0.25)
        train_c = 2.0 * torch.sin(2 * 3.14 * train_x + 0.25)

        training_data = pd.DataFrame(
            {"x": train_x.numpy(), "y": train_y.numpy(), "c": train_c}
        )

        generator.add_data(training_data)

        # save generator config to file
        gen_dump = generator.to_json(serialize_torch=True)
        options = json.loads(gen_dump)

        with open("test.yml", "w") as f:
            yaml.dump(options, f)

        # load generator config from file
        with open("test.yml", "r") as f:
            saved_options_dict = yaml.safe_load(f)

        # create generator from dict
        saved_options_dict["vocs"] = my_vocs.model_dump()
        dump = json.dumps(saved_options_dict)

        reloaded_json = json.loads(dump)
        ExpectedImprovementGenerator.model_validate(reloaded_json)

        loaded_generator = ExpectedImprovementGenerator.model_validate_json(dump)
        assert isinstance(
            loaded_generator.gp_constructor.covar_modules["y"], ScaleKernel
        )

        # clean up
        os.remove("test.yml")
        os.remove(options["gp_constructor"]["covar_modules"]["y"])

        # specify a periodic kernel for each output (objectives and constraints)
        covar_modules = {
            "y": ScaleKernel(PeriodicKernel()),
            "c": ScaleKernel(PeriodicKernel()),
        }

        gp_constructor = StandardModelConstructor(covar_modules=covar_modules)
        generator = ExpectedImprovementGenerator(
            vocs=my_vocs, gp_constructor=gp_constructor
        )

        # define training data to pass to the generator
        train_x = torch.tensor((0.2, 0.5, 0.6))
        train_y = 5.0 * torch.cos(2 * 3.14 * train_x + 0.25)
        train_c = 2.0 * torch.sin(2 * 3.14 * train_x + 0.25)

        training_data = pd.DataFrame(
            {"x": train_x.numpy(), "y": train_y.numpy(), "c": train_c}
        )

        generator.add_data(training_data)

        # save generator config to file
        options = json.loads(generator.to_json(serialize_torch=True))

        with open("test.yml", "w") as f:
            yaml.dump(options, f)

        # load generator config from file
        with open("test.yml", "r") as f:
            saved_options = yaml.safe_load(f)

        # create generator from file
        saved_options["vocs"] = my_vocs.model_dump()
        loaded_generator = ExpectedImprovementGenerator.model_validate_json(
            json.dumps(saved_options)
        )
        for name, val in loaded_generator.gp_constructor.covar_modules.items():
            assert isinstance(val, ScaleKernel)

        # clean up
        os.remove("test.yml")
        for name in my_vocs.output_names:
            os.remove(options["gp_constructor"]["covar_modules"][name])

    def test_train_model(self):
        # tests to make sure that models created by StandardModelConstructor class
        # match by-hand botorch SingleTaskGP modules

        test_vocs = deepcopy(TEST_VOCS)
        test_data = deepcopy(TEST_DATA)

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
            gp_constructor = StandardModelConstructor(covar_modules=test_covar1)
            constructed_model = gp_constructor.build_model_from_vocs(
                test_vocs, test_data
            ).models[0]

            # build initial model explicitly for comparison
            train_X = torch.cat(
                [
                    torch.tensor(test_data[v]).reshape(-1, 1)
                    for v in test_vocs.variable_names
                ],
                dim=1,
            )
            train_Y = torch.tensor(test_data["y1"]).reshape(-1, 1)
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

            # TODO: fix test
            # test_pts = torch.tensor(
            #    pd.DataFrame(
            #        TEST_VOCS_BASE.random_inputs(5, include_constants=False)
            #    ).to_numpy()
            # )

            # with torch.no_grad():
            #    constructed_prediction = constructed_model.posterior(test_pts).mean
            #    benchmark_prediction = benchmark_model.posterior(test_pts).mean

            # assert torch.allclose(
            #    constructed_prediction, benchmark_prediction, rtol=1e-3
            # )

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
        gp_constructor = StandardModelConstructor(covar_modules=covar_module_dict)

        # construct BAX generator
        generator = ExpectedImprovementGenerator(
            vocs=vocs, gp_constructor=gp_constructor
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
            vocs=vocs, gp_constructor=gp_constructor
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

    def test_unused_modules_warning(self):
        test_vocs = deepcopy(TEST_VOCS)
        test_data = deepcopy(TEST_DATA)

        # test unused covariance or mean module
        kwargs_covar = {"covar_modules": {"faulty_output_name": PeriodicKernel()}}
        kwargs_mean = {"mean_modules": {"faulty_output_name": ConstantMean()}}

        for kwargs in [kwargs_covar, kwargs_mean]:
            gp_constructor = StandardModelConstructor(**kwargs)
            generator = ExpectedImprovementGenerator(
                vocs=test_vocs, gp_constructor=gp_constructor
            )
            generator.add_data(test_data)
            with pytest.warns(UserWarning):
                _ = generator.train_model()

    def test_heteroskedastic(self):
        test_data = deepcopy(TEST_DATA)
        test_vocs = deepcopy(TEST_VOCS)

        gp_constructor = StandardModelConstructor()
        model = gp_constructor.build_model_from_vocs(test_vocs, test_data)
        assert isinstance(model.models[0], SingleTaskGP)

        # test with variance data
        test_data["y1_var"] = test_data["y1"] * 0.1
        model = gp_constructor.build_model_from_vocs(test_vocs, test_data)

        assert isinstance(model.models[0], XoptHeteroskedasticSingleTaskGP)

    def test_custom_noise_prior(self):
        test_data = deepcopy(TEST_DATA)
        test_vocs = deepcopy(TEST_VOCS)

        noise_prior = GammaPrior(1.0, 1000.0)

        gp_constructor = StandardModelConstructor(custom_noise_prior=noise_prior)
        model = gp_constructor.build_model_from_vocs(test_vocs, test_data)

        # check if created models have the correct noise priors
        assert model.models[0].likelihood.noise_covar.noise_prior.rate == 1000.0
        assert model.models[0].likelihood.noise_covar.noise_prior.concentration == 1.0

        assert model.models[1].likelihood.noise_covar.noise_prior.rate == 1000.0
        assert model.models[1].likelihood.noise_covar.noise_prior.concentration == 1.0

    def test_model_caching(self):
        test_data = deepcopy(TEST_DATA)
        test_vocs = deepcopy(TEST_VOCS)

        constructor = StandardModelConstructor()

        constructor.build_model(
            test_vocs.variable_names, test_vocs.output_names, test_data
        )

        # cache model
        old_model = constructor.build_model_from_vocs(test_vocs, test_data)

        state = deepcopy(constructor._hyperparameter_store)
        assert torch.equal(
            old_model.models[0].covar_module.raw_lengthscale,
            state["models.0.covar_module.raw_lengthscale"],
        )

        # add data and use the cached model hyperparameters
        constructor.use_cached_hyperparameters = True
        test_data = pd.concat(
            (
                test_data,
                pd.DataFrame(
                    {
                        **{
                            f"x{i + 1}": [0.2, 0.1]
                            for i in range(test_vocs.n_variables)
                        },
                        **{
                            f"y{i + 1}": [0.1, 0.2]
                            for i in range(test_vocs.n_objectives)
                        },
                    }
                ),
            )
        )

        def compare_dicts_with_tensors(dict1, dict2):
            # Check if both have the same keys
            if dict1.keys() != dict2.keys():
                return False

            # Compare each value
            for key in dict1:
                val1, val2 = dict1[key], dict2[key]
                # Check if both are tensors
                if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                    if not torch.equal(val1, val2):  # Use torch.equal for tensors
                        print(f"Tensor mismatch at key {key}: {val1} vs {val2}")
                        return False
                else:
                    # Fall back to standard equality for non-tensors
                    if val1 != val2:
                        print(f"Value mismatch at key {key}: {val1} vs {val2}")
                        return False

            return True

        constructor.train_model = False
        new_model = constructor.build_model_from_vocs(test_vocs, test_data)
        assert compare_dicts_with_tensors(
            new_model.state_dict(), old_model.state_dict()
        )

        constructor.train_model = True
        new_model = constructor.build_model_from_vocs(test_vocs, test_data)
        with pytest.raises(AssertionError):
            assert compare_dicts_with_tensors(
                new_model.state_dict(), old_model.state_dict()
            )

        # test error handling - should raise a warning that hyperparameters were not
        # used
        constructor = StandardModelConstructor()
        constructor.use_cached_hyperparameters = True

        with pytest.raises(RuntimeWarning):
            constructor.build_model(
                test_vocs.variable_names, test_vocs.output_names, test_data
            )

    @pytest.fixture(autouse=True)
    def clean_up(self):
        yield
        files = ["test.yml", "covar_modules_y.pt", "covar_modules_c.pt"]
        for f in files:
            if os.path.exists(f):
                os.remove(f)


class TestBatchedModelConstructor:
    def test_train_model_batch(self):
        # tests to make sure that models created by BatchedModelConstructor class
        # generate correct shapes
        test_vocs = deepcopy(TEST_VOCS)
        test_data = deepcopy(TEST_DATA)

        gp_constructor = BatchedModelConstructor()

        train_X, train_Y, train_Yvar = get_training_data_batched(
            input_names=test_vocs.variable_names,
            outcome_names=test_vocs.output_names,
            data=test_data,
            batch_mode=False,
        )
        assert train_X.shape == torch.Size([10, test_vocs.n_variables])
        assert train_X[0, 0] == test_data.loc[0, "x1"]
        assert train_Y.shape == torch.Size([10, test_vocs.n_outputs])
        assert train_Y[0, 0] == test_data.loc[0, "y1"]
        assert train_Y[0, 1] == test_data.loc[0, "c1"]

        train_X, train_Y, train_Yvar = get_training_data_batched(
            input_names=test_vocs.variable_names,
            outcome_names=test_vocs.output_names,
            data=test_data,
            batch_mode=True,
        )
        assert train_X.shape == torch.Size(
            [test_vocs.n_outputs, 10, test_vocs.n_variables]
        )
        assert train_X[0, 0, 0] == train_X[1, 0, 0] == test_data.loc[0, "x1"]
        assert train_Y.shape == torch.Size([test_vocs.n_outputs, 10, 1])
        assert train_Y[0, 0, 0] == test_data.loc[0, "y1"]
        assert train_Y[1, 0, 0] == test_data.loc[0, "c1"]

        model = gp_constructor.build_model_from_vocs(test_vocs, test_data)

        assert isinstance(model, SingleTaskGP)
        assert model._aug_batch_shape == torch.Size([2])
        assert model.covar_module.raw_lengthscale.shape == torch.Size([2, 1, 3])

    def test_train_model_batch_compare(self):
        # test to verify that BatchedModelConstructor produces same results as
        # StandardModelConstructor with multiple SingleTaskGP models
        test_vocs = deepcopy(TEST_VOCS)
        test_data = deepcopy(TEST_DATA)
        verbose = False

        torch.manual_seed(42)
        gp_constructor = BatchedModelConstructor(train_model=False)
        model_single = gp_constructor.build_model_from_vocs(test_vocs, test_data)
        ls = model_single.covar_module.raw_lengthscale.shape
        if test_vocs.n_outputs > 1:
            assert ls == torch.Size([test_vocs.n_outputs, 1, test_vocs.n_variables])
        else:
            assert ls == torch.Size([1, test_vocs.n_variables])

        constructor = StandardModelConstructor(train_model=False)
        model_list = constructor.build_model_from_vocs(test_vocs, test_data)
        assert model_list.models[0].covar_module.raw_lengthscale.shape == torch.Size(
            [1, test_vocs.n_variables]
        )

        list_ls = [
            model_list.models[i].covar_module.raw_lengthscale
            for i in range(test_vocs.n_outputs)
        ]
        batch_ls = model_single.covar_module.raw_lengthscale
        for i in range(test_vocs.n_outputs):
            assert torch.allclose(list_ls[i], batch_ls[i, ...], rtol=0, atol=1e-3)
        if verbose:
            print([(p, p.shape) for p in model_list.parameters()])
            print("------------------------------")
            print([(p, p.shape) for p in model_single.parameters()])

        mll = ExactMarginalLogLikelihood(model_single.likelihood, model_single)

        isingle = 0
        single_losses = np.zeros(1)

        def cb_single(parameters, result):
            nonlocal isingle
            if verbose:
                print(f"SINGLE{isingle} {result} ")
                for k, v in parameters.items():
                    print(f"  {k}: {super(type(v), v).__repr__()}")
            isingle += 1
            single_losses[0] = result.fval

        fit_gpytorch_mll(mll, optimizer_kwargs={"callback": cb_single})

        list_losses = np.zeros(len(model_list.models))
        for i, ml in enumerate(model_list.models):
            ilist = 0

            def cb_list(parameters, result):
                nonlocal ilist
                if verbose:
                    print(f"LIST{ilist} {result} ")
                    for k, v in parameters.items():
                        print(f"  {k}: {super(type(v), v).__repr__()}")
                ilist += 1
                list_losses[i] = result.fval

            mll = ExactMarginalLogLikelihood(ml.likelihood, ml)
            fit_gpytorch_mll(mll, optimizer_kwargs={"callback": cb_list})

        if verbose:
            print(f"Single losses: {single_losses}")
            print(f"List losses: {list_losses}, sum {list_losses.sum()}")
        assert np.isclose(list_losses.sum(), single_losses.sum(), rtol=0.0, atol=1e-4)

        list_ls = [
            model_list.models[i].covar_module.raw_lengthscale
            for i in range(test_vocs.n_outputs)
        ]
        batch_ls = model_single.covar_module.raw_lengthscale
        with pytest.raises(AssertionError):
            # Hyperparameters do not match after training (they are close-ish)
            # This is because L-BFGS-B terminates at different places for individual
            # models, but the loss is summed for single model so we run until shared
            # stopping criterion
            for i in range(test_vocs.n_outputs):
                assert torch.allclose(list_ls[i], batch_ls[i, ...], rtol=0, atol=1e-3)

    def test_train_model_batch_compare_adam(self):
        test_vocs = deepcopy(TEST_VOCS)
        test_data = deepcopy(TEST_DATA)

        gp_constructor = BatchedModelConstructor(train_model=False)
        model_single = gp_constructor.build_model_from_vocs(test_vocs, test_data)
        ls = model_single.covar_module.raw_lengthscale.shape
        if test_vocs.n_outputs > 1:
            assert ls == torch.Size([test_vocs.n_outputs, 1, test_vocs.n_variables])
        else:
            assert ls == torch.Size([1, test_vocs.n_variables])

        constructor = StandardModelConstructor(train_model=False)
        model_list = constructor.build_model_from_vocs(test_vocs, test_data)
        assert model_list.models[0].covar_module.raw_lengthscale.shape == torch.Size(
            [1, test_vocs.n_variables]
        )

        list_ls = [
            model_list.models[i].covar_module.raw_lengthscale
            for i in range(test_vocs.n_outputs)
        ]
        batch_ls = model_single.covar_module.raw_lengthscale
        for i in range(test_vocs.n_outputs):
            assert torch.allclose(list_ls[i], batch_ls[i, ...], rtol=0, atol=1e-3)
        print([(p, p.shape) for p in model_list.parameters()])
        print("------------------------------")
        print([(p, p.shape) for p in model_single.parameters()])

        optimizer_single = torch.optim.Adam(model_single.parameters(), lr=0.1)
        mll = ExactMarginalLogLikelihood(model_single.likelihood, model_single)
        optimizer_list = []
        mll_list = []
        for ml in model_list.models:
            optimizer_list.append(torch.optim.Adam(ml.parameters(), lr=0.1))
            mll_list.append(ExactMarginalLogLikelihood(ml.likelihood, ml))

        for i in range(10):
            optimizer_single.zero_grad()
            output = model_single(model_single.train_inputs[0])
            loss = -mll(output, model_single.train_targets).sum()
            loss.backward()
            optimizer_single.step()
            single_loss = float(loss.item())
            print(f"Single: {loss.item()}")
            total_list_loss = 0.0
            for j, ml in enumerate(model_list.models):
                optimizer_list[j].zero_grad()
                output = ml(ml.train_inputs[0])
                loss = -mll_list[j](output, ml.train_targets)
                loss.backward()
                optimizer_list[j].step()
                print(f"List {j}: {loss.item()}")
                total_list_loss += loss.item()
            print(f"List total: {total_list_loss}")
            assert np.isclose(total_list_loss, single_loss, rtol=0.0, atol=1e-8)
        list_ls = [
            model_list.models[i].covar_module.raw_lengthscale
            for i in range(test_vocs.n_outputs)
        ]
        batch_ls = model_single.covar_module.raw_lengthscale
        for i in range(test_vocs.n_outputs):
            assert torch.allclose(list_ls[i], batch_ls[i, ...], rtol=0, atol=1e-3)

    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_batched_basic(self, use_cuda):
        test_data = deepcopy(TEST_DATA)
        test_vocs = deepcopy(TEST_VOCS)
        constructor = BatchedModelConstructor()

        model = constructor.build_model_from_vocs(
            test_vocs, test_data, device=device_map[use_cuda]
        )
        assert isinstance(model, SingleTaskGP)
        verify_state_device(model, device=device_map[use_cuda])

    def test_batched_transform_inputs_false(self):
        test_data = deepcopy(TEST_DATA)
        test_vocs = deepcopy(TEST_VOCS)

        constructor = BatchedModelConstructor(transform_inputs=False)
        model = constructor.build_model_from_vocs(test_vocs, test_data)
        assert not hasattr(model, "input_transform")

    def test_batched_transform_inputs_dict_raises(self):
        test_data = deepcopy(TEST_DATA)
        test_vocs = deepcopy(TEST_VOCS)

        constructor = BatchedModelConstructor(transform_inputs={"c1": False})
        with pytest.raises(AttributeError):
            constructor.build_model_from_vocs(test_vocs, test_data)

    def test_batched_model_w_nans(self):
        test_data = deepcopy(TEST_DATA)
        test_vocs = deepcopy(TEST_VOCS)

        # add nans to outputs -- batched drops rows with any NaN across all outputs
        test_data.loc[5, "y1"] = np.nan
        test_data.loc[6, "c1"] = np.nan
        test_data.loc[7, "c1"] = np.nan

        constructor = BatchedModelConstructor()
        model = constructor.build_model_from_vocs(test_vocs, test_data)
        # rows 5, 6, 7 dropped -> 7 remaining
        # SingleTaskGP unrolls to (n_outputs, n_samples, n_vars)
        n_out = test_vocs.n_outputs
        assert model.train_inputs[0].shape == torch.Size(
            [n_out, 7, test_vocs.n_variables]
        )

        # add nans to inputs
        test_data2 = deepcopy(TEST_DATA)
        test_data2.loc[5, "x1"] = np.nan

        model2 = constructor.build_model_from_vocs(test_vocs, test_data2)
        assert model2.train_inputs[0].shape == torch.Size(
            [n_out, 9, test_vocs.n_variables]
        )

    def test_batched_model_caching(self):
        test_data = deepcopy(TEST_DATA)
        test_vocs = deepcopy(TEST_VOCS)

        constructor = BatchedModelConstructor()
        constructor.build_model_from_vocs(test_vocs, test_data)

        old_model = constructor.build_model_from_vocs(test_vocs, test_data)
        state = deepcopy(constructor._hyperparameter_store)
        assert torch.equal(
            old_model.covar_module.raw_lengthscale,
            state["covar_module.raw_lengthscale"],
        )

        # add data and use cached hyperparameters
        constructor.use_cached_hyperparameters = True
        extra = pd.DataFrame(
            {
                **{f"x{i + 1}": [0.2, 0.1] for i in range(test_vocs.n_variables)},
                **{f"y{i + 1}": [0.1, 0.2] for i in range(test_vocs.n_objectives)},
            }
        )
        test_data_extended = pd.concat((test_data, extra))

        def compare_dicts_with_tensors(dict1, dict2):
            if dict1.keys() != dict2.keys():
                return False
            for key in dict1:
                val1, val2 = dict1[key], dict2[key]
                if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                    if not torch.equal(val1, val2):
                        return False
                elif val1 != val2:
                    return False
            return True

        constructor.train_model = False
        new_model = constructor.build_model_from_vocs(test_vocs, test_data_extended)
        assert compare_dicts_with_tensors(
            new_model.state_dict(), old_model.state_dict()
        )

        constructor.train_model = True
        new_model = constructor.build_model_from_vocs(test_vocs, test_data_extended)
        with pytest.raises(AssertionError):
            assert compare_dicts_with_tensors(
                new_model.state_dict(), old_model.state_dict()
            )

        # empty hyperparameter store should raise
        constructor = BatchedModelConstructor()
        constructor.use_cached_hyperparameters = True
        with pytest.raises(RuntimeWarning):
            constructor.build_model_from_vocs(test_vocs, test_data)

    def test_batched_covar_modules_multiple_raises(self):
        test_vocs = deepcopy(TEST_VOCS)
        test_data = deepcopy(TEST_DATA)

        constructor = BatchedModelConstructor(
            covar_modules={
                "y1": ScaleKernel(PeriodicKernel()),
                "c1": ScaleKernel(PeriodicKernel()),
            }
        )
        with pytest.raises(ValueError, match="cannot be specified individually"):
            constructor.build_model_from_vocs(test_vocs, test_data)

    def test_batched_mean_modules_multiple_raises(self):
        test_vocs = deepcopy(TEST_VOCS)
        test_data = deepcopy(TEST_DATA)

        constructor = BatchedModelConstructor(
            mean_modules={"y1": ConstantMean(), "c1": ConstantMean()}
        )
        with pytest.raises(ValueError, match="cannot be specified individually"):
            constructor.build_model_from_vocs(test_vocs, test_data)

    def test_batched_single_covar_module(self):
        test_vocs = deepcopy(TEST_VOCS)
        test_data = deepcopy(TEST_DATA)

        constructor = BatchedModelConstructor(
            covar_modules={"all": ScaleKernel(PeriodicKernel())}
        )
        model = constructor.build_model_from_vocs(test_vocs, test_data)
        assert isinstance(model, SingleTaskGP)
        assert isinstance(model.covar_module.base_kernel, PeriodicKernel)

    def test_batched_single_mean_module(self):
        test_vocs = deepcopy(TEST_VOCS)
        test_data = deepcopy(TEST_DATA)

        constructor = BatchedModelConstructor(mean_modules={"all": ConstantMean()})
        model = constructor.build_model_from_vocs(test_vocs, test_data)
        assert isinstance(model, SingleTaskGP)
        assert isinstance(model.mean_module, ConstantMean)

    def test_batched_heteroskedastic(self):
        test_data = deepcopy(TEST_DATA)
        test_vocs = deepcopy(TEST_VOCS)

        constructor = BatchedModelConstructor()

        # partial variance columns — not allowed
        test_data_partial = deepcopy(TEST_DATA)
        test_data_partial["y1_var"] = test_data_partial["y1"].abs() * 0.1
        with pytest.raises(ValueError, match="all or none"):
            constructor.build_model_from_vocs(test_vocs, test_data_partial)

        # without variance — uses custom GaussianLikelihood from get_likelihood()
        model_no_var = constructor.build_model_from_vocs(test_vocs, test_data)
        assert isinstance(model_no_var, SingleTaskGP)
        assert isinstance(model_no_var.likelihood, GaussianLikelihood)

        # all variance columns — uses FixedNoiseGaussianLikelihood
        test_data_full = deepcopy(TEST_DATA)
        test_data_full["y1_var"] = test_data_full["y1"].abs() * 0.1
        test_data_full["c1_var"] = test_data_full["c1"].abs() * 0.1
        model_with_var = constructor.build_model_from_vocs(test_vocs, test_data_full)
        assert isinstance(model_with_var, SingleTaskGP)
        assert not isinstance(model_with_var.likelihood, GaussianLikelihood)

    def test_batched_empty_data(self):
        test_vocs = deepcopy(TEST_VOCS)
        # create empty dataframe with correct columns
        empty_data = pd.DataFrame(columns=["x1", "x2", "x3", "y1", "c1", "constant1"])

        constructor = BatchedModelConstructor()
        with pytest.raises(ValueError, match="no data found"):
            constructor.build_model_from_vocs(test_vocs, empty_data)
