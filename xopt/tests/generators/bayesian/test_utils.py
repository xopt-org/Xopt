import time
from copy import deepcopy

import gpytorch
import numpy as np
import pandas as pd
import pytest
import torch
from botorch.acquisition import UpperConfidenceBound

from xopt import Evaluator, Xopt
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
from xopt.generators.bayesian.objectives import create_mobo_objective
from xopt.generators.bayesian.utils import (
    compute_hypervolume_and_pf,
    torch_compile_acqf,
    torch_compile_gp_model,
    torch_trace_acqf,
    torch_trace_gp_model,
    interpolate_points,
    validate_turbo_controller_base,
)
from xopt.resources.benchmarking import time_call
from xopt.resources.testing import TEST_VOCS_BASE, xtest_callable
from xopt.vocs import random_inputs, VOCS
from xopt.generators.bayesian.turbo import (
    OptimizeTurboController,
    SafetyTurboController,
)


cuda_combinations = [False] if not torch.cuda.is_available() else [False, True]
device_map = {False: torch.device("cpu"), True: torch.device("cuda:0")}


class TestUtils:
    def test_mobo_objective(self):
        test_vocs_copy = deepcopy(TEST_VOCS_BASE)
        test_vocs_copy.objectives["y2"] = "MAXIMIZE"
        obj = create_mobo_objective(test_vocs_copy)

        # test large sample shape
        test_samples = torch.randn(3, 4, 5, 3).double()
        output = obj(test_samples)
        assert output.shape == torch.Size([3, 4, 5, 2])

        # test to make sure values are correct - minimize axis should be negated
        test_samples = torch.rand(5, 4, 3)
        output = obj(test_samples)
        assert torch.allclose(output[..., 1], test_samples[..., 1])
        assert torch.allclose(output[..., 0], -test_samples[..., 0])

    def test_compute_hypervolume_and_pf(self):
        # Empty input
        X = torch.empty((0, 2))
        Y = torch.empty((0, 2))
        reference_point = torch.tensor([0.0, 0.0])
        pf_X, pf_Y, _, hv = compute_hypervolume_and_pf(X, Y, reference_point)
        assert pf_X is None
        assert pf_Y is None
        assert hv == 0.0

        # No Pareto front (all points worse than reference)
        X = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        Y = -1 * torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        reference_point = torch.tensor([0.0, 0.0])
        pf_X, pf_Y, _, hv = compute_hypervolume_and_pf(X, Y, reference_point)
        assert pf_X is None
        assert pf_Y is None
        assert hv == 0.0

        # Simple 2D case, two non-dominated points
        X = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        Y = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        reference_point = torch.tensor([0.0, 0.0])
        pf_X, pf_Y, _, hv = compute_hypervolume_and_pf(X, Y, reference_point)
        assert pf_X.shape[1] == 2
        assert pf_Y.shape[1] == 2
        assert hv > 0

        # Reference point is on the Pareto front
        X = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        Y = -1 * torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        reference_point = torch.tensor([0.0, 0.0])
        pf_X, pf_Y, _, hv = compute_hypervolume_and_pf(X, Y, reference_point)
        assert pf_X is None
        assert pf_Y is None
        assert hv == 0.0

        # 3D case
        X = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [3.0, 2.0, 1.0]])
        Y = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [3.0, 2.0, 1.0]])
        reference_point = torch.tensor([0.0, 0.0, 0.0])
        pf_X, pf_Y, _, hv = compute_hypervolume_and_pf(X, Y, reference_point)
        assert pf_X.shape[1] == 3
        assert pf_Y.shape[1] == 3
        assert hv > 0

    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_model_jit(self, use_cuda):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.constraints = {}
        evaluator = Evaluator(function=xtest_callable)
        gen = UpperConfidenceBoundGenerator(
            vocs=vocs,
        )
        gen.use_cuda = use_cuda
        X = Xopt(generator=gen, evaluator=evaluator, vocs=vocs)
        gen = X.generator
        X.random_evaluate(200)
        gen.train_model()
        X.random_evaluate(5000)
        gen.gp_constructor.use_cached_hyperparameters = True
        gen.train_model()
        gen.model.eval()

        def get_model():
            return deepcopy(gen.model.models[0])

        t1 = time.perf_counter()
        model_jit = torch_trace_gp_model(
            get_model(), gen.vocs, gen.tkwargs, posterior=False, batch_size=500
        ).to(device_map[use_cuda])
        t2 = time.perf_counter()
        print(f"JIT compile: {t2 - t1:.4f} seconds")

        t1 = time.perf_counter()
        model_jit_posterior = torch_trace_gp_model(
            get_model(), gen.vocs, gen.tkwargs, batch_size=500
        ).to(device_map[use_cuda])
        t2 = time.perf_counter()
        print(f"JIT posterior compile: {t2 - t1:.4f} seconds")

        x_grid = torch.tensor(
            pd.DataFrame(
                random_inputs(gen.vocs, 500, include_constants=False)
            ).to_numpy()
        )
        x_grid = x_grid.to(device_map[use_cuda])

        m = get_model()
        t, values1 = time_call(lambda: m(x_grid), 3)
        t = np.array(t)
        print(f"Original time: {t[1:].mean():.6f}  +- {t[1:].std():.6f}")

        m = get_model()
        t, values1 = time_call(lambda: m.posterior(x_grid), 3)
        t = np.array(t)
        print(f"Original posterior time: {t[1:].mean():.6f}  +- {t[1:].std():.6f}")

        t, values1 = time_call(lambda: model_jit(x_grid), 3)
        t = np.array(t)
        print(f"JIT time: {t[1:].mean():.6f}  +- {t[1:].std():.6f}")

        t, values1 = time_call(lambda: model_jit_posterior(x_grid), 3)
        t = np.array(t)
        print(f"JIT posterior time: {t[1:].mean():.6f}  +- {t[1:].std():.6f}")

    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_model_compile(self, use_cuda):
        # For inductor + windows any, MSVC 2022 build tools are required
        # For inductor + windows GPU, triton-windows package is required
        # For inductor + linux GPU, triton package is required
        print(f"{torch._dynamo.list_backends()=}")
        print(f"{torch._dynamo.is_dynamo_supported()=}")
        torch._dynamo.reset()
        torch._dynamo.config.cache_size_limit = 32

        evaluator = Evaluator(function=xtest_callable)
        gen = UpperConfidenceBoundGenerator(
            vocs=TEST_VOCS_BASE,
        )
        gen.use_cuda = use_cuda
        gen.numerical_optimizer.n_restarts = 2
        gen.n_monte_carlo_samples = 4
        X = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)
        X.random_evaluate(100)
        for _ in range(1):
            X.step()

        gen = X.generator
        model = gen.train_model().models[0]

        t1 = time.perf_counter()
        model_compile = torch_compile_gp_model(
            gen.train_model().models[0], gen.vocs, gen.tkwargs
        ).to(device_map[use_cuda])
        t2 = time.perf_counter()
        print(f"Compile: {t2 - t1:.4f} seconds")

        t1 = time.perf_counter()
        model_compile_reduce_overhead = torch_compile_gp_model(
            gen.train_model().models[0], gen.vocs, gen.tkwargs, mode="reduce-overhead"
        ).to(device_map[use_cuda])
        t2 = time.perf_counter()
        print(f"Compile RO: {t2 - t1:.4f} seconds")

        t1 = time.perf_counter()
        model_compile_max_autotune = torch_compile_gp_model(
            gen.train_model().models[0], gen.vocs, gen.tkwargs, mode="max-autotune"
        ).to(device_map[use_cuda])
        t2 = time.perf_counter()
        print(f"Compile AT: {t2 - t1:.4f} seconds")

        def fmodel(m, x):
            mvn = m.posterior(x)
            return mvn.mean, mvn.variance

        t1 = time.perf_counter()
        model_jit = torch_trace_gp_model(
            gen.train_model().models[0],
            gen.vocs,
            gen.tkwargs,
        ).to(device_map[use_cuda])
        t2 = time.perf_counter()
        print(f"JIT trace: {t2 - t1:.4f} seconds")

        x_grid = torch.tensor(
            pd.DataFrame(
                random_inputs(gen.vocs, 20, include_constants=False)
            ).to_numpy()
        )
        x_grid = x_grid.to(device_map[use_cuda])

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            t1, values1 = time_call(lambda: fmodel(model, x_grid), 10)
            t1 = np.array(t1)

            t2, values2 = time_call(lambda: model_jit(x_grid), 10)
            t2 = np.array(t2)

            t3, values3 = time_call(lambda: fmodel(model_compile, x_grid), 10)
            t3 = np.array(t3)

            t4, values4 = time_call(
                lambda: fmodel(model_compile_reduce_overhead, x_grid), 10
            )
            t4 = np.array(t4)

            t5, values5 = time_call(
                lambda: fmodel(model_compile_max_autotune, x_grid), 10
            )
            t5 = np.array(t5)

            print(f"Original time: {t1} seconds")
            print(f"JIT time: {t2} seconds")
            print(f"Compiled time: {t3} seconds")
            print(f"Compiled RO time: {t4} seconds")
            print(f"Compiled AT time: {t5} seconds")
            print(f"Avg: {t1[1:].mean():.6f}  +- {t1[1:].std():.6f}")
            print(f"Avg: {t2[1:].mean():.6f}  +- {t2[1:].std():.6f}")
            print(f"Avg: {t3[1:].mean():.6f}  +- {t3[1:].std():.6f}")
            print(f"Avg: {t4[1:].mean():.6f}  +- {t4[1:].std():.6f}")
            print(f"Avg: {t5[1:].mean():.6f}  +- {t5[1:].std():.6f}")

        for v1, v2, v3 in zip(values1, values2, values3):
            m1, var1 = v1
            m2, var2 = v2
            m3, var3 = v3
            assert torch.allclose(m1, m2, rtol=0), "JIT model output mismatch"
            assert torch.allclose(var1, var2, rtol=0), "JIT model variance mismatch"
            assert torch.allclose(m1, m3, rtol=0), "Compiled model output mismatch"
            assert torch.allclose(var1, var3, rtol=0), (
                "Compiled model variance mismatch"
            )

    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_acqf_compile(self, use_cuda):
        print(f"{torch._dynamo.list_backends()=}")
        print(f"{torch._dynamo.is_dynamo_supported()=}")
        torch._dynamo.reset()
        torch._dynamo.config.cache_size_limit = 32
        # enable to see all the problems with compilation
        # torch._logging.set_logs(graph_breaks=True, recompiles=True)
        # torch._dynamo.config.capture_scalar_outputs = True

        evaluator = Evaluator(function=xtest_callable)
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.constraints = {}
        gen = UpperConfidenceBoundGenerator(
            vocs=vocs,
        )
        gen.use_cuda = use_cuda
        gen.numerical_optimizer.n_restarts = 3
        gen.n_monte_carlo_samples = 4
        X = Xopt(generator=gen, evaluator=evaluator, vocs=vocs)
        X.random_evaluate(200)
        for _ in range(1):
            X.step()

        gen = X.generator

        def make_acqf():
            return gen.get_acquisition(gen.train_model())

        model = make_acqf()

        acqf = make_acqf().to(device_map[use_cuda])
        t1 = time.perf_counter()
        model_compile = torch_compile_acqf(acqf, gen.vocs, gen.tkwargs).to(
            device_map[use_cuda]
        )
        t2 = time.perf_counter()
        print(f"Compile: {t2 - t1:.4f} seconds")

        acqf = make_acqf().to(device_map[use_cuda])
        t1 = time.perf_counter()
        model_compile_reduce_overhead = torch_compile_acqf(
            acqf, gen.vocs, gen.tkwargs, mode="reduce-overhead", verify=True
        ).to(device_map[use_cuda])
        t2 = time.perf_counter()
        print(f"Compile RO: {t2 - t1:.4f} seconds")

        acqf = make_acqf().to(device_map[use_cuda])
        t1 = time.perf_counter()
        model_compile_max_autotune = torch_compile_acqf(
            acqf, gen.vocs, gen.tkwargs, mode="max-autotune", verify=True
        ).to(device_map[use_cuda])
        t2 = time.perf_counter()
        print(f"Compile AT: {t2 - t1:.4f} seconds")

        acqf = make_acqf().to(device_map[use_cuda])
        t1 = time.perf_counter()
        model_jit = torch_trace_acqf(acqf, gen.vocs, gen.tkwargs).to(
            device_map[use_cuda]
        )
        t2 = time.perf_counter()
        print(f"JIT trace: {t2 - t1:.4f} seconds")

        def fmodel(m, x):
            return m(x)

        # batch x 1 x d
        x_grid = torch.tensor(
            pd.DataFrame(random_inputs(gen.vocs, 4, include_constants=False)).to_numpy()
        ).unsqueeze(-2)
        x_grid = x_grid.to(device_map[use_cuda])

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            t1, values1 = time_call(lambda: fmodel(model, x_grid), 10)
            t1 = np.array(t1)

            t2, values2 = time_call(lambda: fmodel(model_jit, x_grid), 10)
            t2 = np.array(t2)

            t3, values3 = time_call(lambda: fmodel(model_compile, x_grid), 10)
            t3 = np.array(t3)

            t4, values4 = time_call(
                lambda: fmodel(model_compile_reduce_overhead, x_grid), 10
            )
            t4 = np.array(t4)

            t5, values5 = time_call(
                lambda: fmodel(model_compile_max_autotune, x_grid), 10
            )
            t5 = np.array(t5)

            print(f"Original time: {t1} seconds")
            print(f"JIT time: {t2} seconds")
            print(f"Compiled time: {t3} seconds")
            print(f"Compiled RO time: {t4} seconds")
            print(f"Compiled AT time: {t5} seconds")
            print(f"Original Avg: {t1[1:].mean():.6f}  +- {t1[1:].std():.6f}")
            print(f"JIT Avg: {t2[1:].mean():.6f}  +- {t2[1:].std():.6f}")
            print(f"Compiled Avg: {t3[1:].mean():.6f}  +- {t3[1:].std():.6f}")
            print(f"Compiled RO Avg: {t4[1:].mean():.6f}  +- {t4[1:].std():.6f}")
            print(f"Compiled AT Avg: {t5[1:].mean():.6f}  +- {t5[1:].std():.6f}")

        for v1, v2, v3 in zip(values1, values2, values3):
            m1 = v1
            m2 = v2
            m3 = v3
            assert torch.allclose(m1, m2, rtol=1e-5)
            assert torch.allclose(m1, m3, rtol=1e-5)

    def test_trace_gp_model_model_list_error(self):
        from botorch.models import ModelListGP

        vocs = deepcopy(TEST_VOCS_BASE)
        model = ModelListGP()
        with pytest.raises(ValueError):
            torch_trace_gp_model(model, vocs, {}, posterior=True)

    def test_compile_gp_model_model_list_error(self):
        from botorch.models import ModelListGP

        vocs = deepcopy(TEST_VOCS_BASE)
        model = ModelListGP()
        with pytest.raises(ValueError):
            torch_compile_gp_model(model, vocs, {}, posterior=True)

    def test_torch_trace_acqf(self):
        evaluator = Evaluator(function=xtest_callable)
        gen = UpperConfidenceBoundGenerator(
            vocs=TEST_VOCS_BASE,
        )
        gen.numerical_optimizer.n_restarts = 2
        gen.n_monte_carlo_samples = 4
        X = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)
        X.random_evaluate(100)
        for _ in range(1):
            X.step()

        gen = X.generator
        model = gen.train_model().models[0]
        acq = UpperConfidenceBound(model, beta=0.1)
        tkwargs = {"device": torch.device("cpu"), "dtype": torch.double}
        traced_acq = torch_trace_acqf(acq, gen.vocs, tkwargs)
        assert isinstance(traced_acq, torch.jit.ScriptModule)
        # Check output shape matches original
        rand_point = random_inputs(gen.vocs)[0]
        rand_vec = torch.stack(
            [rand_point[k] * torch.ones(1) for k in gen.vocs.variable_names], dim=1
        ).to(**tkwargs)
        test_x = rand_vec.unsqueeze(-2)
        orig_out = acq(test_x)
        traced_out = traced_acq(test_x)
        assert torch.allclose(orig_out, traced_out, rtol=1e-6)

    def test_interpolate_points_invalid_rows(self):
        # Create a DataFrame with more than two rows
        df_invalid = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]})

        # Expect a ValueError when calling interpolate_points
        with pytest.raises(
            ValueError, match="Input DataFrame must have exactly two rows."
        ):
            interpolate_points(df_invalid)

    def test_validate_turbo_controller_base_all_branches(self):
        class DummyInfo:
            data = {
                "vocs": VOCS(
                    variables={"x": [0, 1]},
                    objectives={"y": "MAXIMIZE"},
                    constraints={},
                    observables=["y"],
                )
            }

        info = DummyInfo()
        valid_types = [OptimizeTurboController, SafetyTurboController]

        # String input: "optimize"
        result = validate_turbo_controller_base("optimize", valid_types, info)
        assert isinstance(result, OptimizeTurboController)

        # String input: invalid
        with pytest.raises(ValueError, match="not found"):
            validate_turbo_controller_base("invalid", valid_types, info)

        # Dict input: valid
        result = validate_turbo_controller_base(
            {"name": "OptimizeTurboController"}, valid_types, info
        )
        assert isinstance(result, OptimizeTurboController)

        # Dict input: missing name
        with pytest.raises(ValueError, match="needs to have a `name` attribute"):
            validate_turbo_controller_base({}, valid_types, info)

        # Dict input: invalid name
        with pytest.raises(ValueError, match="not found"):
            validate_turbo_controller_base(
                {"name": "InvalidController"}, valid_types, info
            )

        # Add constraints to info
        info.data["vocs"].constraints = {"c1": ["LESS_THAN", 0.5]}
        # String input: "safety"
        result = validate_turbo_controller_base("safety", valid_types, info)
        assert isinstance(result, SafetyTurboController)

        # Wrong type: not a valid controller
        class DummyController:
            pass

        with pytest.raises(ValueError, match="not allowed for this generator"):
            validate_turbo_controller_base(DummyController(), valid_types, info)
