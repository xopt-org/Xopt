import io
import os
import tempfile
import torch
import numpy as np
import pandas as pd
import pytest
import yaml
import inspect
from pydantic import BaseModel, Field
from xopt.pydantic import (
    recursive_serialize,
    recursive_deserialize,
    orjson_dumps,
    orjson_dumps_custom,
    orjson_dumps_except_root,
    orjson_loads,
    process_torch_module,
    encode_torch_module,
    decode_torch_module,
    XoptBaseModel,
    remove_none_values,
    get_descriptions_defaults,
    ObjLoaderMinimal,
    ObjLoader,
    NormalExecutor,
    SignatureModel,
    validate_and_compose_signature,
)


class DummyModel(BaseModel):
    a: int = 1
    b: str = "foo"
    c: None = None


class DummyTorchModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)


def test_recursive_serialize_and_deserialize():
    d = {
        "a": 1,
        "b": {"c": 2},
        "d": np.array([1, 2]),
        "e": set([1, 2]),
        "f": pd.DataFrame({"x": [1, 2]}),
    }
    ser = recursive_serialize(d.copy())
    assert isinstance(ser["d"], list)
    assert isinstance(ser["e"], list)
    assert isinstance(ser["f"], dict)
    # test recursive_deserialize
    d2 = {"a": 1, "b": {"c": 2}, "dtype": "torch.float32"}
    deser = recursive_deserialize(d2.copy())
    assert deser["dtype"] == torch.float32


def test_orjson_dumps_and_loads():
    m = DummyModel()
    s = orjson_dumps(m)
    assert isinstance(s, str)
    loaded = orjson_loads(s)
    assert loaded["a"] == 1
    # test custom
    s2 = orjson_dumps_custom(m, default=lambda x: str(x))
    assert isinstance(s2, str)
    # except root
    d = orjson_dumps_except_root(m)
    assert isinstance(d, dict)


def test_process_and_encode_decode_torch_module():
    mod = DummyTorchModule()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = process_torch_module(mod, os.path.join(tmpdir, "testmod"))
        assert os.path.exists(path)
    # encode/decode
    encoded = encode_torch_module(mod)
    decoded = decode_torch_module("base64:" + encoded)
    assert isinstance(decoded, torch.nn.Module)


def test_xoptbasemodel_to_json_yaml(tmp_path):
    class M(XoptBaseModel):
        a: int = 1

    m = M()
    assert isinstance(m.to_json(), str)
    assert isinstance(m.json(), str)
    assert isinstance(m.yaml(), str)
    # test from_dict
    m2 = M.from_dict({"a": 2})
    assert m2.a == 2
    # test from_yaml
    yaml_str = yaml.dump({"a": 3})
    m3 = M.from_yaml(io.StringIO(yaml_str))
    assert m3.a == 3
    # test from_file
    file = tmp_path / "test.yaml"
    file.write_text(yaml_str)
    m4 = M.from_file(str(file))
    assert m4.a == 3
    # test file not found
    with pytest.raises(OSError):
        M.from_file("nonexistent.yaml")


def test_remove_none_values():
    d = {"a": 1, "b": None, "c": {"d": None, "e": 2}, "f": [None, 3]}
    cleaned = remove_none_values(d)
    assert "b" not in cleaned
    assert "d" not in cleaned["c"]
    assert cleaned["f"] == [3]


def test_get_descriptions_defaults():
    class M(XoptBaseModel):
        """desc"""

        a: int = 1

    m = M()
    desc = get_descriptions_defaults(m)
    assert "a" in desc


def test_get_descriptions_defaults_nested():
    class Inner(XoptBaseModel):
        """inner desc"""

        x: int = 42

    class Outer(XoptBaseModel):
        """outer desc"""

        inner: Inner = Inner()
        y: float = 3.14

    o = Outer()
    desc = get_descriptions_defaults(o)
    # Should recurse into inner and get its description dict
    assert "inner" in desc
    assert isinstance(desc["inner"], dict)
    assert "x" in desc["inner"]
    assert "y" in desc


def test_get_descriptions_defaults_object_type():
    class DummyCallable:
        pass

    class M(XoptBaseModel):
        """desc"""

        a: DummyCallable = Field(DummyCallable(), description="callable field")

    m = M()
    desc = get_descriptions_defaults(m)
    # Should handle object/callable type
    assert desc["a"][0] == "callable field"


def test_objloader_minimal():
    class Dummy:
        pass

    loader = ObjLoaderMinimal[Dummy]()
    assert loader.object_type == Dummy


def test_signaturemodel_build():
    class S(SignatureModel):
        args: list = [1, 2]
        kwarg_order: list = ["x"]
        x: int = 3

    s = S()
    args, kwargs = s.build(4, x=5)
    assert args == [4, 2]  # positional overwrite
    assert kwargs["x"] == 5


def test_baseexecutor_and_normalexecutor():
    class DummyExec:
        def submit(self, fn, *args, **kwargs):
            return "submitted"

        def map(self, fn, *args, **kwargs):
            return ["mapped"]

        def shutdown(self):
            return "shutdown"

    loader = ObjLoader[DummyExec]()  # <-- Use ObjLoader, not ObjLoaderMinimal
    be = NormalExecutor[DummyExec](loader=loader, executor=DummyExec())
    assert be.submit(lambda x: x, 1) == "submitted"
    assert be.map(lambda x: x, [1, 2]) == ["mapped"]
    be.shutdown()


def test_validate_and_compose_signature_tuple_and_empty():
    def fn_tuple(x=(1, 2)):
        pass

    model = validate_and_compose_signature(fn_tuple)
    # Should create a field with type tuple and default None
    assert hasattr(model, "x")
    assert model.model_fields["x"].annotation is tuple
    assert model.model_fields["x"].default is None

    def fn_empty(x=inspect.Parameter.empty):
        pass

    model = validate_and_compose_signature(fn_empty)
    # Should create a field with type inspect.Parameter.empty and default inspect.Parameter.empty
    assert hasattr(model, "x")
    assert model.model_fields["x"].annotation == inspect.Parameter.empty
    assert model.model_fields["x"].default == inspect.Parameter.empty

    def fn_none(x=None):
        pass

    model = validate_and_compose_signature(fn_none)
    # Should create a field with type inspect.Parameter.empty and default None
    assert hasattr(model, "x")
    assert model.model_fields["x"].annotation == inspect.Parameter.empty
    assert model.model_fields["x"].default is None

    def fn_int(x=5):
        pass

    model = validate_and_compose_signature(fn_int)
    # Should create a field with type int and default 5
    assert hasattr(model, "x")
    assert model.model_fields["x"].annotation is int
    assert model.model_fields["x"].default == 5


def test_objloader_validate_all_loader_variants():
    class Dummy:
        pass

    # Loader not in values: should create CallableModel with Dummy as callable
    loader = ObjLoader[Dummy]()
    assert loader.object_type == Dummy
    assert isinstance(loader.loader, type(loader.loader))
    # Loader is already a CallableModel
    loader2 = ObjLoader[Dummy](loader=loader.loader)
    assert loader2.object_type == Dummy
    assert isinstance(loader2.loader, type(loader.loader))
    # Loader is a dict with 'callable' key
    loader3 = ObjLoader[Dummy](loader={"callable": Dummy})
    assert loader3.object_type == Dummy
    assert isinstance(loader3.loader, type(loader.loader))
    # Loader is a dict without 'callable' key
    loader4 = ObjLoader[Dummy](loader={})
    assert loader4.object_type == Dummy
    assert isinstance(loader4.loader, type(loader.loader))

    # Loader with wrong callable type should raise ValueError
    class Other:
        pass

    with pytest.raises(ValueError):
        ObjLoader[Dummy](loader={"callable": Other})
