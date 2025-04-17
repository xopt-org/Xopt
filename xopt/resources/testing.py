from typing import Any

import numpy as np
import pandas as pd
import torch

from xopt.vocs import VOCS


def xtest_callable(input_dict: dict, a=0) -> dict:
    assert isinstance(input_dict, dict)
    x1 = input_dict["x1"]
    x2 = input_dict["x2"]

    assert "constant1" in input_dict

    y1 = x2
    c1 = x1
    return {"y1": y1, "c1": c1}


def xtest_callable_mo(input_dict: dict) -> dict:
    assert isinstance(input_dict, dict)
    x1 = input_dict["x1"]
    x2 = input_dict["x2"]

    assert "constant1" in input_dict

    y1 = x2
    y2 = x1
    c1 = x1
    return {"y1": y1, "y2": y2, "c1": c1}


def verify_state_device(state: dict, device: torch.device):
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            if v.device != device:
                raise ValueError(
                    f"Tensor {k} is on device {v.device}, expected {device}"
                )


def recursive_torch_device_scan(
    obj: Any, visited: set, device: torch.device, depth=0, verbose=False
):
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
    # print("Checking objective")
    objective = gen._get_objective()
    state = objective.state_dict()
    verify_state_device(state, device)

    if gen.data is not None and not gen.data.empty:
        # print("Checking model state dict")
        model = gen.train_model(gen.data)
        state = model.state_dict()
        verify_state_device(state, device)

        # print("Checking sampler state dict")
        state = gen._get_sampler(model).state_dict()
        verify_state_device(state, device)

        # print("Checking acquisition state dict")
        acqf = gen.get_acquisition(model)
        state = acqf.state_dict()
        verify_state_device(state, device)

    # print("Recursing into generator")
    visited = set()
    recursive_torch_device_scan(gen, visited, device)


TEST_VOCS_BASE = VOCS(
    **{
        "variables": {"x1": [0, 1.0], "x2": [0, 10.0]},
        "objectives": {"y1": "MINIMIZE"},
        "constraints": {"c1": ["GREATER_THAN", 0.5]},
        "constants": {"constant1": 1.0},
    }
)
TEST_VOCS_BASE_MO = TEST_VOCS_BASE.model_copy()
TEST_VOCS_BASE_MO.objectives["y2"] = "MINIMIZE"

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
