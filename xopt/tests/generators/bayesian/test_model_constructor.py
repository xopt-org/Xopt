import json
import os
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
    StandardModelConstructor,
)
from xopt.generators.bayesian.utils import get_training_data_batched
from xopt.resources.testing import (
    TEST_VOCS_BASE,
    TEST_VOCS_DATA,
    verify_state_device,
)
from xopt.vocs import VOCS

cuda_combinations = [False] if not torch.cuda.is_available() else [False, True]
device_map = {False: torch.device("cpu"), True: torch.device("cuda:0")}


class TestModelConstructor:
    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_standard(self, use_cuda):
        test_data = deepcopy(TEST_VOCS_DATA)
        test_vocs = deepcopy(TEST_VOCS_BASE)
        constructor = StandardModelConstructor()

        model = constructor.build_model_from_vocs(
            test_vocs, test_data, device=device_map[use_cuda]
        )
        verify_state_device(model, device=device_map[use_cuda])

    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_standard_adam(self, use_cuda):
        test_data = deepcopy(TEST_VOCS_DATA)
        test_vocs = deepcopy(TEST_VOCS_BASE)
        constructor = StandardModelConstructor(method="adam")
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
        test_data = deepcopy(TEST_VOCS_DATA)
        test_vocs = deepcopy(TEST_VOCS_BASE)

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
        test_data = deepcopy(TEST_VOCS_DATA)
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.observables = ["y1"]

        constructor = StandardModelConstructor()

        constructor.build_model(
            test_vocs.variable_names, test_vocs.output_names, test_data
        )

        model = constructor.build_model_from_vocs(test_vocs, test_data)
        assert model.num_outputs == 2

    def test_custom_model(self):
        test_data = deepcopy(TEST_VOCS_DATA)
        test_vocs = deepcopy(TEST_VOCS_BASE)

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
        test_data = deepcopy(TEST_VOCS_DATA)
        test_vocs = deepcopy(TEST_VOCS_BASE)
        constructor = StandardModelConstructor()

        # add nans to ouputs
        test_data.loc[5, "y1"] = np.nan
        test_data.loc[6, "c1"] = np.nan
        test_data.loc[7, "c1"] = np.nan

        model = constructor.build_model_from_vocs(test_vocs, test_data)

        assert model.train_inputs[0][0].shape == torch.Size([9, test_vocs.n_variables])
        assert model.train_inputs[1][0].shape == torch.Size([8, test_vocs.n_variables])

        # add nans to inputs
        test_data2 = deepcopy(TEST_VOCS_DATA)
        test_data2.loc[5, "x1"] = np.nan

        model2 = constructor.build_model_from_vocs(test_vocs, test_data2)
        assert model2.train_inputs[0][0].shape == torch.Size([9, test_vocs.n_variables])

        # add nans to both
        test_data3 = deepcopy(TEST_VOCS_DATA)
        test_data3.loc[5, "x1"] = np.nan
        test_data3.loc[7, "c1"] = np.nan

        model3 = constructor.build_model_from_vocs(test_vocs, test_data3)
        assert model3.train_inputs[0][0].shape == torch.Size([9, test_vocs.n_variables])
        assert model3.train_inputs[1][0].shape == torch.Size([8, test_vocs.n_variables])

    def test_model_w_same_data(self):
        test_data = deepcopy(TEST_VOCS_DATA)
        test_vocs = deepcopy(TEST_VOCS_BASE)
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

        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_data = deepcopy(TEST_VOCS_DATA)

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

    def test_train_model_batch(self):
        # tests to make sure that models created by BatchedModelConstructor class
        # generate correct shapes
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_data = deepcopy(TEST_VOCS_DATA)

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
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_data = deepcopy(TEST_VOCS_DATA)

        torch.manual_seed(42)
        gp_constructor = BatchedModelConstructor(train_model=False)
        model_single = gp_constructor.build_model_from_vocs(test_vocs, test_data)
        ls = model_single.covar_module.raw_lengthscale.shape
        if test_vocs.n_outputs > 1:
            assert ls == torch.Size([test_vocs.n_outputs, 1, test_vocs.n_variables])
        else:
            assert ls == torch.Size([1, test_vocs.n_variables])

        torch.manual_seed(42)
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

        mll = ExactMarginalLogLikelihood(model_single.likelihood, model_single)
        fit_gpytorch_mll(mll)

        for ml in model_list.models:
            mll = ExactMarginalLogLikelihood(ml.likelihood, ml)
            fit_gpytorch_mll(mll)

        list_ls = [
            model_list.models[i].covar_module.raw_lengthscale
            for i in range(test_vocs.n_outputs)
        ]
        batch_ls = model_single.covar_module.raw_lengthscale
        with pytest.raises(AssertionError):
            # Something is wrong - hyperparameters do not match after training (they are close)
            # This works in gpytorch, need to investigate
            for i in range(test_vocs.n_outputs):
                assert torch.allclose(list_ls[i], batch_ls[i, ...], rtol=0, atol=1e-3)

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
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_data = deepcopy(TEST_VOCS_DATA)

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
        test_data = deepcopy(TEST_VOCS_DATA)
        test_vocs = deepcopy(TEST_VOCS_BASE)

        gp_constructor = StandardModelConstructor()
        model = gp_constructor.build_model_from_vocs(test_vocs, test_data)
        assert isinstance(model.models[0], SingleTaskGP)

        # test with variance data
        test_data["y1_var"] = test_data["y1"] * 0.1
        model = gp_constructor.build_model_from_vocs(test_vocs, test_data)

        assert isinstance(model.models[0], XoptHeteroskedasticSingleTaskGP)

    def test_custom_noise_prior(self):
        test_data = deepcopy(TEST_VOCS_DATA)
        test_vocs = deepcopy(TEST_VOCS_BASE)

        noise_prior = GammaPrior(1.0, 1000.0)

        gp_constructor = StandardModelConstructor(custom_noise_prior=noise_prior)
        model = gp_constructor.build_model_from_vocs(test_vocs, test_data)

        # check if created models have the correct noise priors
        assert model.models[0].likelihood.noise_covar.noise_prior.rate == 1000.0
        assert model.models[0].likelihood.noise_covar.noise_prior.concentration == 1.0

        assert model.models[1].likelihood.noise_covar.noise_prior.rate == 1000.0
        assert model.models[1].likelihood.noise_covar.noise_prior.concentration == 1.0

    def test_model_caching(self):
        test_data = deepcopy(TEST_VOCS_DATA)
        test_vocs = deepcopy(TEST_VOCS_BASE)

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
                        "x1": [0.2, 0.1],
                        "x2": [0.2, 0.1],
                        "y1": [0.2, 0.1],
                        "y2": [0.2, 0.1],
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
                        return False
                else:
                    # Fall back to standard equality for non-tensors
                    if val1 != val2:
                        return False

            return True

        new_model = constructor.build_model_from_vocs(test_vocs, test_data)
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
