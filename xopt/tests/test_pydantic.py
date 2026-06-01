import inspect
import io
import json
import os
import tempfile
from functools import partial
from types import FunctionType, MethodType
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, field_validator
from pydantic.json import custom_pydantic_encoder

from xopt.pydantic import (
    JSON_ENCODERS,
    CallableModel,
    NormalExecutor,
    ObjLoader,
    ObjLoaderMinimal,
    SignatureModel,
    XoptBaseModel,
    decode_torch_module,
    encode_torch_module,
    get_callable_from_string,
    get_descriptions_defaults,
    orjson_dumps,
    orjson_dumps_custom,
    orjson_dumps_except_root,
    orjson_loads,
    process_torch_module,
    recursive_deserialize,
    recursive_serialize,
    remove_none_values,
    validate_and_compose_signature,
)


def misc_fn(x, y=1, *args, **kwargs):
    pass


class MiscClass:
    @staticmethod
    def misc_static_method(x, y=1, *args, **kwargs):
        return

    @classmethod
    def misc_cls_method(cls, x, y=1, *args, **kwargs):
        return cls

    def misc_method(self, x, y=1, *args, **kwargs):
        return


class TestJsonEncoders:
    misc_class = MiscClass()

    @pytest.mark.parametrize(
        ("fn",),
        [
            (misc_fn,),
            pytest.param(misc_class.misc_method, marks=pytest.mark.xfail(strict=True)),
            (misc_class.misc_static_method,),
            pytest.param(
                misc_class.misc_cls_method, marks=pytest.mark.xfail(strict=True)
            ),
        ],
    )
    def test_function_type(self, fn):
        encoder = {FunctionType: JSON_ENCODERS[FunctionType]}
        json_encoder = partial(custom_pydantic_encoder, encoder)

        serialized = json.dumps(fn, default=json_encoder)
        loaded = json.loads(serialized)
        callable_from_str = get_callable_from_string(loaded)

        assert fn == callable_from_str

    @pytest.mark.parametrize(
        ("fn",),
        [
            pytest.param(
                misc_class.misc_static_method, marks=pytest.mark.xfail(strict=True)
            ),
            pytest.param(misc_fn, marks=pytest.mark.xfail(strict=True)),
            (misc_class.misc_method,),
            pytest.param(
                misc_class.misc_cls_method, marks=pytest.mark.xfail(strict=True)
            ),
        ],
    )
    def test_method_type(self, fn):
        encoder = {MethodType: JSON_ENCODERS[MethodType]}
        json_encoder = partial(custom_pydantic_encoder, encoder)

        serialized = json.dumps(fn, default=json_encoder)
        loaded = json.loads(serialized)
        callable = get_callable_from_string(loaded, bind=self.misc_class)

        assert fn == callable

    @pytest.mark.parametrize(
        ("fn",),
        [
            (misc_class.misc_static_method,),
            (misc_fn,),
            (misc_class.misc_method,),
            (misc_class.misc_cls_method,),
        ],
    )
    def test_full_encoder(self, fn):
        json_encoder = partial(custom_pydantic_encoder, JSON_ENCODERS)
        serialized = json.dumps(fn, default=json_encoder)
        loaded = json.loads(serialized)

        get_callable_from_string(loaded)


class TestSignatureValidateAndCompose:
    misc_class = MiscClass()

    @pytest.mark.parametrize(
        ("args", "kwargs"),
        [
            pytest.param((5, 2, 1), {"x": 2}, marks=pytest.mark.xfail(strict=True)),
            pytest.param((), ({"y": 2}), marks=pytest.mark.xfail(strict=True)),
            pytest.param((2,), ({"x": 2}), marks=pytest.mark.xfail(strict=True)),
            ((), ({"x": 2})),
            ((), {}),
        ],
    )
    def test_validate_kwarg_only(self, args, kwargs):
        def run(*, x: int = 4):
            pass

        signature_model = validate_and_compose_signature(run, *args, **kwargs)
        assert all(
            [kwargs[kwarg] == getattr(signature_model, kwarg) for kwarg in kwargs]
        )
        # run

        args, kwargs = signature_model.build()

        run(*args, **kwargs)

    @pytest.mark.parametrize(
        ("args", "kwargs"),
        [
            pytest.param(
                (
                    5,
                    3,
                    2,
                ),
                {"x": 1},
                marks=pytest.mark.xfail(strict=True),
            ),
            ((2, 1, 0), {}),
            ((), {}),
        ],
    )
    def test_validate_var_positional(self, args, kwargs):
        def run(*args):
            pass

        signature_model = validate_and_compose_signature(run, *args, **kwargs)
        args, kwargs = signature_model.build()
        assert len(kwargs) == 0
        assert len(args) == len(args)

        # run
        run(*args)

    @pytest.mark.parametrize(
        ("args", "kwargs"),
        [
            pytest.param((5,), {"x": 2}, marks=pytest.mark.xfail(strict=True)),
            ((), {"x": 2, "y": 3}),
            pytest.param((), {}, marks=pytest.mark.xfail(strict=True)),
            (
                (
                    2,
                    4,
                ),
                {},
            ),
            ((2,), {"y": 4, "extra": True}),
            ((2,), {"y": 4}),
            ((2,), {"y": 4, "z": 3}),
        ],
    )
    def test_validate_full_sig(self, args, kwargs):
        def run(x, y, z=4, *args, **kwargs):
            pass

        signature_model = validate_and_compose_signature(run, *args, **kwargs)
        args, kwargs = signature_model.build()

        # run
        run(*args, **kwargs)

    @pytest.mark.parametrize(
        ("args", "kwargs"),
        [
            pytest.param((5, 1), {"y": 2}, marks=pytest.mark.xfail(strict=True)),
            (
                (
                    2,
                    4,
                ),
                {},
            ),
            ((5,), {"y": 2}),
        ],
    )
    def test_validate_classmethod(self, args, kwargs):
        signature_model = validate_and_compose_signature(
            self.misc_class.misc_cls_method, *args, **kwargs
        )
        args, kwargs = signature_model.build()
        self.misc_class.misc_cls_method(*args, **kwargs)

    @pytest.mark.parametrize(
        ("args", "kwargs"),
        [
            pytest.param((5, 1), {"y": 2}, marks=pytest.mark.xfail(strict=True)),
            (
                (
                    2,
                    4,
                ),
                {},
            ),
            ((5,), {"y": 2}),
        ],
    )
    def test_validate_staticmethod(self, args, kwargs):
        signature_model = validate_and_compose_signature(
            self.misc_class.misc_static_method, *args, **kwargs
        )
        args, kwargs = signature_model.build()
        self.misc_class.misc_static_method(*args, **kwargs)

    @pytest.mark.parametrize(
        ("args", "kwargs"),
        [
            pytest.param((5, 1), {"y": 2}, marks=pytest.mark.xfail(strict=True)),
            (
                (
                    2,
                    4,
                ),
                {},
            ),
            ((5,), {"y": 2}),
        ],
    )
    def test_validate_bound_method(self, args, kwargs):
        signature_model = validate_and_compose_signature(
            self.misc_class.misc_method, *args, **kwargs
        )

        args, kwargs = signature_model.build()

        self.misc_class.misc_method(*args, **kwargs)


class TestCallableModel:
    misc_class = MiscClass()

    @pytest.mark.parametrize(
        ("fn", "args", "kwargs"),
        [
            (misc_fn, (5,), {"y": 2}),
            (misc_class.misc_cls_method, (5,), {"y": 2}),
            (misc_class.misc_static_method, (5,), {"y": 2}),
            pytest.param(
                misc_class.misc_method,
                (5,),
                {"y": 2},
                marks=pytest.mark.xfail(strict=True),
            ),
        ],
    )
    def test_construct_callable(self, fn, args, kwargs):
        json_encoder = partial(custom_pydantic_encoder, JSON_ENCODERS)
        serialized = json.dumps(fn, default=json_encoder)
        loaded = json.loads(serialized)

        callable = CallableModel(callable=loaded)
        callable(*args, **kwargs)

    @pytest.mark.parametrize(
        ("fn", "args", "kwargs"),
        [
            pytest.param(misc_fn, (5,), {"y": 2}, marks=pytest.mark.xfail(strict=True)),
            pytest.param(
                misc_class.misc_cls_method,
                (5,),
                {"y": 2},
                marks=pytest.mark.xfail(strict=True),
            ),
            pytest.param(
                misc_class.misc_static_method,
                (5,),
                {"y": 2},
                marks=pytest.mark.xfail(strict=True),
            ),
            (misc_class.misc_method, (5,), {"y": 2}),
        ],
    )
    def test_bound_callables(self, fn, args, kwargs):
        json_encoder = partial(custom_pydantic_encoder, JSON_ENCODERS)
        serialized = json.dumps(fn, default=json_encoder)
        loaded = json.loads(serialized)

        callable = CallableModel(callable=loaded, bind=self.misc_class)
        callable(*args, **kwargs)


class TestObjLoader:
    misc_class_loader_type = ObjLoader[MiscClass]

    def test_class_loader(self):
        loader = self.misc_class_loader_type()
        assert loader.object_type == MiscClass

    def test_load_model(self):
        loader = self.misc_class_loader_type()
        misc_obj = loader.load()
        assert isinstance(misc_obj, (MiscClass,))

    def test_serialize_loader(self):
        loader = self.misc_class_loader_type()

        json_encoder = partial(custom_pydantic_encoder, JSON_ENCODERS)
        serialized = json.dumps(loader, default=json_encoder)

        # self.misc_class_loader_type.parse_raw(serialized)
        # This works in 2.2+ as it should
        self.misc_class_loader_type.model_validate_json(serialized)


# tests to verify v2 behavior remains same (for things that changed from v1)


class DummyObj:
    pass


class Dummy(BaseModel):
    default_obj: DummyObj = Field(DummyObj())
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("default_obj")
    def validate_obj(cls, value):
        assert isinstance(value, DummyObj)
        return value


# Test subclass model resolution order
# we want behavior like v1 had https://github.com/pydantic/pydantic/issues/1932
class Parent(BaseModel):
    a1: str = "a1"


class Child1(Parent):
    name: str = "child1"


class Child2(Parent):
    name: str = "child2"


class Container(BaseModel):
    obj: SerializeAsAny[Optional[Parent]] = Field(None)
    obj2: SerializeAsAny[Optional[Union[Child1, Child2, Parent]]] = Field(None)


class TestPydanticInitialization:
    def test_object_init(self):
        d = Dummy()
        assert isinstance(d.default_obj, DummyObj)

    def test_subclass_init(self):
        c1 = Container()
        print("c1", c1.model_dump())
        c2 = Container(obj=Child2())
        print("c2", c2.model_dump())
        # doesn't resolve child1
        c3 = Container(**{"obj": {"a1": "a1", "name": "child1"}})
        print(type(c3.obj), type(c3.obj2), c3)
        # works
        c4 = Container(**{"obj2": {"a1": "a1", "name": "child1"}})
        print(type(c4.obj), type(c4.obj2), c4)


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

    ser = recursive_serialize({"float": torch.float32, "unserizable": DummyModel()})
    assert ser == {
        "float": "torch.float32",
        "unserizable": "xopt.tests.test_pydantic.DummyModel",
    }


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

    # test torch load in XoptBaseModel
    torch.save(torch.nn.Linear(2, 2), tmp_path / "model.pt")
    M.validate_files(yaml.safe_load(str(tmp_path / "model.pt")))


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
        f: Callable = lambda x: x + 1

    m = M()
    desc = get_descriptions_defaults(m)
    assert "a" in desc
    assert "f" in desc

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

    # test serialize object type
    res = loader.serialize_object_type(None)
    assert res is None
    res = loader.serialize_object_type(Dummy)
    assert res == f"{Dummy.__module__}.{Dummy.__name__}"


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
        pass  # pragma: no cover

    model = validate_and_compose_signature(fn_tuple)
    # Should create a field with type tuple and default None
    assert hasattr(model, "x")
    assert model.model_fields["x"].annotation is tuple
    assert model.model_fields["x"].default is None

    def fn_empty(x=inspect.Parameter.empty):
        pass  # pragma: no cover

    model = validate_and_compose_signature(fn_empty)
    # Should create a field with type inspect.Parameter.empty and default inspect.Parameter.empty
    assert hasattr(model, "x")
    assert model.model_fields["x"].annotation == inspect.Parameter.empty
    assert model.model_fields["x"].default == inspect.Parameter.empty

    def fn_none(x=None):
        pass  # pragma: no cover

    model = validate_and_compose_signature(fn_none)
    # Should create a field with type inspect.Parameter.empty and default None
    assert hasattr(model, "x")
    assert model.model_fields["x"].annotation == inspect.Parameter.empty
    assert model.model_fields["x"].default is None

    def fn_int(x=5):
        pass  # pragma: no cover

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

    # test serialization of loader
    for loader in [loader, loader2, loader3, loader4]:
        loader.serialize_json()

    # Loader with wrong callable type should raise ValueError
    class Other:
        pass

    with pytest.raises(ValueError):
        ObjLoader[Dummy](loader={"callable": Other})


def test_objloader_load_store_and_no_store():
    class Dummy:
        def __init__(self):
            self.value = 42

    loader = ObjLoader[Dummy]()
    # Test store=False (should return a new Dummy instance, not store it)
    result1 = loader.load(store=False)
    assert isinstance(result1, Dummy)
    assert loader.object is None  # Should not store
    # Test store=True (should store the Dummy instance)
    result2 = loader.load(store=True)
    assert isinstance(result2, Dummy)
    assert loader.object is result2  # Should store
