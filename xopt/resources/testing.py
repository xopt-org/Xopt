import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from botorch.exceptions import OptimizationWarning
from torch import nn

from xopt import Generator
from xopt.pydantic import remove_none_values
from xopt.vocs import VOCS

# TODO: make a config module like gpytorch has
# A manual flag to trigger verification of torch device in some locations
XOPT_VERIFY_TORCH_DEVICE = True
# Disable the check + softplus in case of negative values in the base acquisition function
XOPT_VERIFY_CONSTRAINED_ACQF_POSITIVE = True


def xtest_callable(input_dict: dict, a=0) -> dict:
    """Single-objective callable test function"""
    assert isinstance(input_dict, dict)
    x1 = input_dict["x1"]
    x2 = input_dict["x2"]

    assert "constant1" in input_dict

    y1 = x2
    c1 = x1
    return {"y1": y1, "c1": c1}


def xtest_callable_mo(input_dict: dict) -> dict:
    """Multi-objective callable test function"""
    assert isinstance(input_dict, dict)
    x1 = input_dict["x1"]
    x2 = input_dict["x2"]

    assert "constant1" in input_dict

    y1 = x2
    y2 = x1
    c1 = x1
    return {"y1": y1, "y2": y2, "c1": c1}


def verify_state_device(module: nn.Module, device: torch.device, prefix=""):
    """
    Verify that all tensors in the module's state_dict are on the specified device.
    """
    state = module.state_dict(keep_vars=True)
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            if v.device != device:
                raise ValueError(
                    f"Tensor {k} from [{module.__class__=}] [{prefix=}] is on device {v.device}, expected {device}"
                )


def recursive_torch_device_scan(
    obj: Any, visited: set, device: torch.device, depth=0, verbose=False
):
    """
    Recursively scan an object for torch tensors and check their device.
    """
    if isinstance(obj, (float, int, str, bool, type, pd.DataFrame)):
        if verbose:
            print(f"{'    ' * depth}skipping basic type {type(obj)}")
        return
    if verbose:
        print(f"{'----' * depth}scanning {type(obj)} at {id(obj)}")
    # attrs_and_slots = {
    #    a: obj.__getattribute__(a) for a in dir(obj) if not a.startswith("__")
    # }
    try:
        attrs_and_slots = obj.__dict__
    except AttributeError:
        if verbose:
            print(f"{'    ' * depth}no dict for {type(obj)} {id(obj)}")
        attrs_and_slots = {}

    try:
        attrs_and_slots.update(**{k: getattr(obj, k) for k in obj.__slots__})
    except AttributeError:
        if verbose:
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
            if verbose:
                print(f"{'    ' * depth}skipping [{k}] {type(v)}")
            continue
        if verbose:
            print(f"{'    ' * depth}checking [{k}] {type(v)}")
        if isinstance(v, torch.nn.Module) or hasattr(v, "state_dict"):
            for name, value in v.state_dict(keep_vars=True).items():
                if isinstance(value, torch.Tensor):
                    assert value.device == device
        elif isinstance(v, list):
            if verbose:
                print(f"{'    ' * depth}recursing into list [{k}] {type(v)}")
            for item in v:
                if verbose:
                    print(f"{'    ' * depth}recursing into list item {type(item)}")
                recursive_torch_device_scan(item, visited, device, depth=depth + 1)
        elif isinstance(v, dict):
            if verbose:
                print(f"{'    ' * depth}recursing into dict [{k}] {type(v)}")
            for kk, vv in v.items():
                if verbose:
                    print(f"{'    ' * depth}recursing into dict item [{kk}] {type(vv)}")
                recursive_torch_device_scan(vv, visited, device, depth=depth + 1)
        elif isinstance(v, set):
            if verbose:
                print(f"{'    ' * depth}recursing into set [{k}] {type(v)}")
            for item in v:
                if verbose:
                    print(f"{'    ' * depth}recursing into set item {type(item)}")
                recursive_torch_device_scan(item, visited, device, depth=depth + 1)
        else:
            # print(f"{'    ' * depth}recursing into {k} {type(v)}")
            recursive_torch_device_scan(v, visited, device, depth=depth + 1)
        visited |= {id(v)}
    if verbose:
        print(
            f"{'++++' * depth}DONE {type(obj)} at {id(obj)} - visited {len(visited)} objects"
        )
    if verbose:
        print(f"{'++++' * depth}")


def check_generator_tensor_locations(gen, device):
    """
    Check that all tensors in the generator are on the specified device.
    """
    # print("Checking objective")
    objective = gen._get_objective()
    verify_state_device(objective, device, "objective")

    if gen.data is not None and not gen.data.empty:
        # print("Checking model state dict")
        model = gen.train_model(gen.data)
        verify_state_device(model, device)

        # print("Checking sampler state dict")
        sampler = gen._get_sampler(model)
        verify_state_device(sampler, device)

        # print("Checking acquisition state dict")
        acqf = gen.get_acquisition(model)
        verify_state_device(acqf, device)

    # print("Recursing into generator")
    visited = set()
    recursive_torch_device_scan(gen, visited, device)


def check_dict_equal(dict1, dict2, excluded_keys=None):
    """
    Compare two dictionaries for equality, ignoring specified keys.
    """
    if excluded_keys is None:
        excluded_keys = []

    if set(dict1.keys()) != set(dict2.keys()):
        raise KeyError(f"Keys in dict1: {dict1.keys()} not in dict2: {dict2.keys()}")

    for key in dict1.keys():
        if key in excluded_keys:
            continue
        if key not in dict2:
            raise KeyError(f"Key {key} not in {dict2}")
        if dict1[key] != dict2[key]:
            raise ValueError(
                f"Key {key} has different values: {dict1[key]} != {dict2[key]}"
            )


def check_dict_allclose(dict1, dict2, excluded_keys=None, rtol=1e-5, atol=1e-8):
    """
    Compare two dictionaries approximately, ignoring specified keys.
    """
    if excluded_keys is None:
        excluded_keys = []

    if set(dict1.keys()) != set(dict2.keys()):
        raise KeyError(f"Keys in dict1: {dict1.keys()} not in dict2: {dict2.keys()}")

    for key in dict1.keys():
        if key in excluded_keys:
            continue
        if isinstance(dict1[key], torch.Tensor):
            v1 = dict1[key].cpu().numpy()
            v2 = dict2[key].cpu().numpy()
            if not np.allclose(v1, v2, rtol=rtol, atol=atol):
                raise ValueError(
                    f"Key {key} has different values: {dict1[key]} != {dict2[key]}"
                )
        elif isinstance(dict1[key], (float, int, np.ndarray)):
            v1 = dict1[key]
            v2 = dict2[key]
            if not np.allclose(v1, v2, rtol=rtol, atol=atol):
                raise ValueError(
                    f"Key {key} has different values: {dict1[key]} != {dict2[key]}"
                )
        else:
            if dict1[key] != dict2[key]:
                raise ValueError(
                    f"Key {key} has different values: {dict1[key]} != {dict2[key]}"
                )


def reload_gen_from_json(gen):
    assert isinstance(gen, Generator)
    gen_class = gen.__class__
    gen_new = gen_class(vocs=gen.vocs, **json.loads(gen.json()))
    gen_new.add_data(gen.data.copy())
    return gen_new


def reload_gen_from_yaml(gen):
    assert isinstance(gen, Generator)
    gen_class = gen.__class__
    gen_new = gen_class(vocs=gen.vocs, **remove_none_values(yaml.safe_load(gen.yaml())))
    gen_new.add_data(gen.data.copy())
    return gen_new


def generate_without_warnings(gen, n, warning_classes: list = None):
    """
    Check that generation/acqf optimization does not silently fail (raising botorch warnings)
    """
    warning_classes = warning_classes or [
        RuntimeWarning,
        OptimizationWarning,
    ]  # UserWarning
    with warnings.catch_warnings(record=True) as w:
        candidates = gen.generate(n)
        bad_warnings = [x for x in w if issubclass(x.category, tuple(warning_classes))]
        if len(bad_warnings) > 0:
            raise RuntimeError(f"Warnings: [{[x.message for x in bad_warnings]}]")
        return candidates


def create_set_options_helper(
    data,
    n_restarts=2,
    n_monte_carlo_samples=4,
):
    def set_options(gen, use_cuda=False, add_data=False):
        gen.use_cuda = use_cuda
        gen.numerical_optimizer.n_restarts = n_restarts
        gen.n_monte_carlo_samples = n_monte_carlo_samples
        if add_data:
            gen.add_data(data)

    return set_options


# Single-objective VOCS with constraints
TEST_VOCS_BASE_DICT: dict[str, Any] = {
    "variables": {"x1": [0, 1.0], "x2": [0, 10.0]},
    "objectives": {"y1": "MINIMIZE"},
    "constraints": {"c1": ["GREATER_THAN", 0.5]},
    "constants": {"constant1": 1.0},
}
TEST_VOCS_BASE = VOCS(**TEST_VOCS_BASE_DICT)

# Multi-objective VOCS with constraints
TEST_VOCS_BASE_MO = TEST_VOCS_BASE.model_copy(deep=True)
TEST_VOCS_BASE_MO.objectives["y2"] = "MINIMIZE"

# Multi-objective VOCS without constraints
TEST_VOCS_BASE_MO_NC = TEST_VOCS_BASE_MO.model_copy(deep=True)
TEST_VOCS_BASE_MO_NC.constraints = {}

# Multi-objective reference point for MOBO
TEST_VOCS_REF_POINT = {"y1": 1.5, "y2": 1.5}

cnames = (
    list(TEST_VOCS_BASE.variables.keys())
    + list(TEST_VOCS_BASE.objectives.keys())
    + list(TEST_VOCS_BASE.constraints.keys())
    + list(TEST_VOCS_BASE.constants.keys())
)

# TODO: figure out why having the range from 0->1 or 0->10 breaks objective test data
#  test
test_init_data = {
    "x1": np.linspace(0.01, 1.0, 10),
    "x2": np.linspace(0.01, 1.0, 10) * 10.0,
    "constant1": 1.0,
}

TEST_VOCS_DATA = pd.DataFrame({**test_init_data, **xtest_callable(test_init_data)})
TEST_VOCS_DATA_MO = pd.DataFrame(
    {**test_init_data, **xtest_callable_mo(test_init_data)}
)

TEST_YAML = """
generator:
    name: random

evaluator:
    function: xopt.resources.testing.xtest_callable
    function_kwargs:
        a: 5

vocs:
    variables:
        x1: [0, 3.14159]
        x2: [0, 3.14159]
    objectives: {y1: MINIMIZE, y2: MINIMIZE}
    constraints:
        c1: [GREATER_THAN, 0]
        c2: ['LESS_THAN', 0.5]
    constants:
        constant1: 1
"""
