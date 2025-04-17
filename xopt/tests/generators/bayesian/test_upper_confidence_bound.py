from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.upper_confidence_bound import (
    UpperConfidenceBoundGenerator,
)
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA, xtest_callable


def verify_same_device(state, device):
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            if v.device != device:
                raise ValueError(
                        f"Tensor {k} is on device {v.device}, expected {device}"
                )


def recursive_torch_scan(obj, visited, device, depth=0):
    if isinstance(obj, (float, int, str, bool, type, pd.DataFrame)):
        print(f"{'    ' * depth}skipping basic type {type(obj)}")
        return
    print(f"{'----' * depth}scanning {type(obj)} at {id(obj)}")
    # attrs_and_slots = {
    #    a: obj.__getattribute__(a) for a in dir(obj) if not a.startswith("__")
    # }
    # attrs_and_slots = {**obj.__dict__ , **{k: getattr(obj, k) for k in obj.__slots__}}
    try:
        attrs_and_slots = obj.__dict__
    except AttributeError:
        print(f"{'    ' * depth}no dict for {type(obj)} {id(obj)}")
        attrs_and_slots = {}

    try:
        attrs_and_slots.update(**{k: getattr(obj, k) for k in obj.__slots__})
    except AttributeError:
        print(f"{'    ' * depth}no slots for {type(obj)} {id(obj)}")
        pass

    for k, v in attrs_and_slots.items():
        if v is None:
            # print(f"{'    ' * depth}skipping None {k}")
            continue
        # if isinstance(v, (float, int, str, bool, type, pd.DataFrame)):
        #    print(f"{'    ' * depth}skipping basic type {k} {v}")
        #    continue
        if id(v) in visited:
            print(f"{'    ' * depth}skipping [{k}] {type(v)}")
            continue
        print(f"{'    ' * depth}checking [{k}] {type(v)}")
        if isinstance(v, torch.nn.Module) or hasattr(v, "state_dict"):
            for name, value in v.state_dict(keep_vars=True).items():
                if isinstance(value, torch.Tensor):
                    assert value.device == device
        elif isinstance(v, list):
            print(f"{'    ' * depth}recursing into list [{k}] {type(v)}")
            for item in v:
                print(f"{'    ' * depth}recursing into list item {type(item)}")
                recursive_torch_scan(item, visited, device, depth=depth + 1)
        elif isinstance(v, dict):
            print(f"{'    ' * depth}recursing into dict [{k}] {type(v)}")
            for kk, vv in v.items():
                print(f"{'    ' * depth}recursing into dict item [{kk}] {type(vv)}")
                recursive_torch_scan(vv, visited, device, depth=depth + 1)
        elif isinstance(v, set):
            print(f"{'    ' * depth}recursing into set [{k}] {type(v)}")
            for item in v:
                print(f"{'    ' * depth}recursing into set item {type(item)}")
                recursive_torch_scan(item, visited, device, depth=depth + 1)
        else:
            # print(f"{'    ' * depth}recursing into {k} {type(v)}")
            recursive_torch_scan(v, visited, device, depth=depth + 1)
        visited |= {id(v)}
    print(f"{'++++' * depth}DONE {type(obj)} at {id(obj)} - visited {len(visited)} objects")
    print(f"{'++++' * depth}")


def check_generator_tensor_locations(gen, device):
    print("Checking objective")
    objective = gen._get_objective()
    state = objective.state_dict()
    for k, v in state.items():
        # print(k, v)
        if isinstance(v, torch.Tensor):
            assert v.device == device

    print("Checking model state dict")
    model = gen.train_model(gen.data)
    state = model.state_dict()
    for k, v in state.items():
        # print(k, v)
        if isinstance(v, torch.Tensor):
            assert v.device == device

    print("Checking sampler state dict")
    state = gen._get_sampler(model).state_dict()
    for k, v in state.items():
        # print(k, v)
        if isinstance(v, torch.Tensor):
            assert v.device == device

    print("Checking acquisition state dict")
    acqf = gen.get_acquisition(model)
    state = acqf.state_dict()
    for k, v in state.items():
        # print(k, v)
        if isinstance(v, torch.Tensor):
            assert v.device == device

    print("Recursing into generator")
    visited = set()
    recursive_torch_scan(gen, visited, device)


class TestUpperConfidenceBoundGenerator:
    def test_init(self):
        ucb_gen = UpperConfidenceBoundGenerator(vocs=TEST_VOCS_BASE)
        ucb_gen.model_dump_json()

        # test init from dict
        UpperConfidenceBoundGenerator(vocs=TEST_VOCS_BASE.dict())

        with pytest.raises(ValueError):
            UpperConfidenceBoundGenerator(
                    vocs=TEST_VOCS_BASE.dict(), log_transform_acquisition_function=True
            )

    def test_generate(self):
        gen = UpperConfidenceBoundGenerator(
                vocs=TEST_VOCS_BASE,
        )
        gen.numerical_optimizer.n_restarts = 1
        gen.n_monte_carlo_samples = 1
        gen.data = TEST_VOCS_DATA

        candidate = gen.generate(1)
        assert len(candidate) == 1

        candidate = gen.generate(2)
        assert len(candidate) == 2

        # test time tracking
        assert isinstance(gen.computation_time, pd.DataFrame)
        assert len(gen.computation_time) == 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        gen = UpperConfidenceBoundGenerator(
                vocs=TEST_VOCS_BASE,
        )

        cuda_device = torch.device("cuda:0")
        gen.use_cuda = True
        gen.numerical_optimizer.n_restarts = 1
        gen.n_monte_carlo_samples = 1
        gen.data = TEST_VOCS_DATA

        candidate = gen.generate(1)
        assert len(candidate) == 1

        check_generator_tensor_locations(gen, cuda_device)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_cpu(self):
        # test that we can ignore cuda even if available
        gen = UpperConfidenceBoundGenerator(
                vocs=TEST_VOCS_BASE,
        )

        cuda_device = torch.device("cpu")
        gen.use_cuda = False
        gen.numerical_optimizer.n_restarts = 1
        gen.n_monte_carlo_samples = 1
        gen.data = TEST_VOCS_DATA

        candidate = gen.generate(1)
        assert len(candidate) == 1

        check_generator_tensor_locations(gen, cuda_device)

    def test_generate_w_overlapping_objectives_constraints(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.constraints = {"y1": ["GREATER_THAN", 0.0]}
        gen = UpperConfidenceBoundGenerator(
                vocs=test_vocs,
        )
        gen.numerical_optimizer.n_restarts = 1
        gen.n_monte_carlo_samples = 1
        gen.data = TEST_VOCS_DATA

        candidate = gen.generate(1)
        assert len(candidate) == 1

    def test_in_xopt(self):
        evaluator = Evaluator(function=xtest_callable)
        gen = UpperConfidenceBoundGenerator(
                vocs=TEST_VOCS_BASE,
        )
        gen.numerical_optimizer.n_restarts = 1
        gen.n_monte_carlo_samples = 1

        X = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)

        # initialize with single initial candidate
        X.random_evaluate(1)

        # now use bayes opt
        for _ in range(1):
            X.step()

    def test_fixed_feature(self):
        # test with fixed feature not in vocs
        gen = UpperConfidenceBoundGenerator(vocs=TEST_VOCS_BASE)
        gen.fixed_features = {"p": 3.0}
        gen.n_monte_carlo_samples = 1
        gen.numerical_optimizer.n_restarts = 1
        data = deepcopy(TEST_VOCS_DATA)
        data["p"] = np.random.rand(len(data))

        gen.add_data(data)
        candidate = gen.generate(1)[0]
        assert candidate["p"] == 3

        # test with fixed feature in vocs
        gen = UpperConfidenceBoundGenerator(vocs=TEST_VOCS_BASE)
        gen.fixed_features = {"x1": 3.0}
        gen.n_monte_carlo_samples = 1
        gen.numerical_optimizer.n_restarts = 1

        gen.add_data(data)
        candidate = gen.generate(1)[0]
        assert candidate["x1"] == 3

    def test_constraints_warning(self):
        with pytest.warns(UserWarning):
            _ = UpperConfidenceBoundGenerator(
                    vocs=TEST_VOCS_BASE,
            )

    def test_negative_acq_values_warning(self):
        X = Xopt.from_yaml(
                """
            generator:
              name: upper_confidence_bound

            evaluator:
              function: xopt.resources.test_functions.sinusoid_1d.evaluate_sinusoid

            vocs:
              variables:
                x1: [0, 6.28]
              constraints:
                c1: [LESS_THAN, 0.0]
              objectives:
                y1: 'MAXIMIZE'
            """
        )
        _ = X.random_evaluate(10, seed=0)
        test_x = torch.linspace(*X.vocs.variables["x1"], 10)
        model = X.generator.train_model(X.data)
        acq = X.generator.get_acquisition(model)
        with pytest.warns(UserWarning):
            _ = acq(test_x.unsqueeze(1).unsqueeze(1))
