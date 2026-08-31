import base64
import gzip
import io
import json
from concurrent.futures import Executor
from typing import Annotated, Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from gpytorch.kernels import RBFKernel
from pydantic import Field

from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.bayesian.objectives import CustomXoptObjective
from xopt.generators.random import RandomGenerator
from xopt.pydantic import (
    CallableModel,
    NormalExecutor,
    ObjLoader,
    XoptBaseModel,
    get_descriptions_defaults,
    validate_and_compose_signature,
)
from xopt.resources.testing import TEST_VOCS_BASE
from xopt.types import (
    CallableRef,
    DataFrameCodec,
    NDArray,
    NDArrayCodec,
    SerializationOptions,
    TorchDType,
    TorchModuleCodec,
    TorchTensor,
    XDataFrame,
    encode_torch_module,
    maybe_decompress,
)


def misc_fn(x=1, y=2):
    return x + y


class MiscClass:
    def __init__(self, value=1):
        self.value = value

    def misc_method(self, x=1):
        return self.value + x


class DummyExecutor(Executor):
    def __init__(self, tag="default"):
        self.tag = tag

    def submit(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return list(map(fn, *iterables))

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.was_shutdown = True


class CodecModel(XoptBaseModel):
    array: NDArray
    tensor: TorchTensor
    dtype: TorchDType
    dataframe: XDataFrame


@pytest.fixture
def codec_model():
    return CodecModel(
        array=np.array([[1.25, 2.5]], dtype=np.float64),
        tensor=torch.tensor([[3.0, 4.0]], dtype=torch.float64),
        dtype=torch.float16,
        dataframe=pd.DataFrame(
            {"x": [1.1234567890123, np.nan], "label": ["a", "b"]},
            index=[3, 7],
        ),
    )


def test_codec_default_shapes_and_python_mode(codec_model):
    dumped = json.loads(codec_model.model_dump_json())
    assert dumped["array"] == [[1.25, 2.5]]
    assert dumped["tensor"] == [[3.0, 4.0]]
    assert dumped["dtype"] == "torch.float16"
    # to_json's default 10-digit precision applies
    assert dumped["dataframe"] == {
        "x": {"3": 1.123456789, "7": None},
        "label": {"3": "a", "7": "b"},
    }

    python_dump = codec_model.model_dump(mode="python")
    assert python_dump["array"] is codec_model.array
    assert python_dump["tensor"] is codec_model.tensor
    assert python_dump["dataframe"] is codec_model.dataframe

    loaded = CodecModel.model_validate_json(codec_model.model_dump_json())
    np.testing.assert_array_equal(loaded.array, codec_model.array)
    torch.testing.assert_close(loaded.tensor, codec_model.tensor)
    assert loaded.dtype is torch.float16
    pd.testing.assert_frame_equal(loaded.dataframe, codec_model.dataframe)


@pytest.mark.parametrize("compression", [None, "gzip", "zstd"])
def test_binary_codec_round_trips(codec_model, compression):
    dumped = codec_model.model_dump_json(
        context={
            "array_mode": "b64",
            "df_mode": "b64",
            "compress": compression,
            "level": 3,
        }
    )
    data = json.loads(dumped)
    assert data["array"].startswith("b64np:")
    assert data["tensor"].startswith("b64pt:")
    assert data["dataframe"].startswith("b64df:")

    raw = base64.b64decode(data["array"].removeprefix("b64np:"))
    expected_magic = {
        None: b"\x93NUMPY",
        "gzip": b"\x1f\x8b",
        "zstd": b"\x28\xb5\x2f\xfd",
    }
    assert raw.startswith(expected_magic[compression])

    loaded = CodecModel.model_validate_json(dumped)
    np.testing.assert_array_equal(loaded.array, codec_model.array)
    torch.testing.assert_close(loaded.tensor, codec_model.tensor)
    # b64df decoding matches dict mode: the integer index is restored
    pd.testing.assert_frame_equal(
        loaded.dataframe, codec_model.dataframe, check_exact=False, atol=1e-9
    )


def test_annotation_defaults_and_context_precedence():
    class ParameterizedModel(XoptBaseModel):
        array: Annotated[np.ndarray, NDArrayCodec(array_mode="b64")]
        dataframe: Annotated[pd.DataFrame, DataFrameCodec(df_mode="b64")]

    model = ParameterizedModel(
        array=np.array([1.0]), dataframe=pd.DataFrame({"x": [1.0]})
    )
    annotated = json.loads(model.model_dump_json())
    assert annotated["array"].startswith("b64np:")
    assert annotated["dataframe"].startswith("b64df:")

    overridden = json.loads(
        model.model_dump_json(context={"array_mode": "list", "df_mode": "dict"})
    )
    assert overridden == {"array": [1.0], "dataframe": {"x": {"0": 1.0}}}

    options = SerializationOptions(array_mode="b64", df_mode="b64")
    object_context = json.loads(model.model_dump_json(context=options))
    assert object_context["array"].startswith("b64np:")

    wrapper_context = json.loads(
        model.to_json(
            array_mode="b64",
            context={"serialization_options": SerializationOptions(array_mode="list")},
        )
    )
    assert wrapper_context["array"] == [1.0]


def test_torch_dtype_generalized_validation():
    class DTypeModel(XoptBaseModel):
        dtype: TorchDType

    for dtype in (
        torch.float32,
        torch.float64,
        torch.float16,
        torch.int64,
        torch.bool,
    ):
        loaded = DTypeModel.model_validate({"dtype": str(dtype)})
        assert loaded.dtype is dtype
    with pytest.raises(ValueError, match="invalid torch dtype"):
        DTypeModel.model_validate({"dtype": "torch.not_a_dtype"})


def test_module_modes_and_sidecar_sanitization(tmp_path, monkeypatch):
    constructor = StandardModelConstructor(covar_modules={"y/bad key": RBFKernel()})

    dropped = json.loads(constructor.to_json())
    assert dropped["covar_modules"] == {}

    inline = json.loads(constructor.to_json(module_mode="inline"))
    assert inline["covar_modules"]["y/bad key"].startswith("b64pt:")
    inline_loaded = StandardModelConstructor.model_validate(inline)
    assert isinstance(inline_loaded.covar_modules["y/bad key"], RBFKernel)

    written = json.loads(constructor.to_json(module_mode="file", file_dir=tmp_path))
    assert written["covar_modules"] == {"y/bad key": "covar_modules_y_bad_key.pt"}
    assert (tmp_path / "covar_modules_y_bad_key.pt").is_file()
    monkeypatch.chdir(tmp_path)
    file_loaded = StandardModelConstructor.model_validate(written)
    assert isinstance(file_loaded.covar_modules["y/bad key"], RBFKernel)


def test_module_inline_legacy_and_prefix_collision_errors():
    class ModuleModel(XoptBaseModel):
        module: Annotated[torch.nn.Module, TorchModuleCodec()]

    legacy = "base64:" + encode_torch_module(torch.nn.Linear(1, 1))[len("b64pt:") :]
    loaded = ModuleModel.model_validate({"module": legacy})
    assert isinstance(loaded.module, torch.nn.Linear)
    with pytest.raises(ValueError, match="invalid b64pt: torch module payload"):
        ModuleModel.model_validate({"module": "base64:AAAA"})
    with pytest.raises(ValueError, match="malformed base64"):
        ModuleModel.model_validate({"module": "b64pt:not-valid!"})
    with pytest.raises(ValueError, match="cannot load torch module"):
        ModuleModel.model_validate({"module": "missing-module.pt"})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("array", "b64np:@@@@", "malformed base64"),
        ("tensor", "b64pt:@@@@", "malformed base64"),
        ("dataframe", "b64df:@@@@", "malformed base64"),
        (
            "array",
            "b64np:" + base64.b64encode(b"not a numpy file").decode(),
            "invalid b64np",
        ),
        (
            "dataframe",
            "b64df:" + base64.b64encode(b"not json").decode(),
            "invalid b64df",
        ),
    ],
)
def test_malformed_binary_payloads(codec_model, field, value, message):
    data = codec_model.model_dump(mode="python")
    data[field] = value
    with pytest.raises(ValueError, match=message):
        CodecModel.model_validate(data)


def test_magic_decompression_and_size_cap():
    raw = b"payload"
    assert maybe_decompress(raw) == raw
    assert maybe_decompress(gzip.compress(raw)) == raw

    import zstandard

    compressed = zstandard.ZstdCompressor().compress(raw)
    assert maybe_decompress(compressed) == raw

    with pytest.raises(ValueError, match="size limit"):
        maybe_decompress(gzip.compress(b"x" * 100), max_size=16)
    with pytest.raises(ValueError, match="size limit"):
        maybe_decompress(zstandard.ZstdCompressor().compress(b"x" * 100), max_size=16)
    # the cap applies to decompression only, not raw passthrough
    assert maybe_decompress(b"x" * 100, max_size=16) == b"x" * 100
    with pytest.raises(ValueError, match="invalid gzip"):
        maybe_decompress(b"\x1f\x8btruncated")
    with pytest.raises(ValueError, match="invalid zstd"):
        maybe_decompress(b"\x28\xb5\x2f\xfdtruncated")


def test_fallback_is_recursive_and_json_native():
    class AnyModel(XoptBaseModel):
        values: dict[str, Any]

    class Unknown:
        pass

    model = AnyModel(
        values={
            "np_scalar": np.int64(2),
            "array": np.array([1, 2]),
            "dtype": torch.float32,
            "tensor": torch.tensor([3, 4]),
            "callable": misc_fn,
            "type": MiscClass,
            "exception": RuntimeError("boom"),
            "unknown": Unknown(),
        }
    )
    dumped = json.loads(model.model_dump_json())
    assert dumped["values"] == {
        "np_scalar": 2,
        "array": [1, 2],
        "dtype": "torch.float32",
        "tensor": [3, 4],
        "callable": f"{__name__}.misc_fn",
        "type": f"{__name__}.MiscClass",
        "exception": "boom",
        "unknown": f"{Unknown.__module__}.{Unknown.__qualname__}",
    }


def test_non_finite_floats_round_trip():
    class NonFiniteModel(XoptBaseModel):
        values: list[float] = [-float("inf"), float("inf"), float("nan")]

    loaded = json.loads(NonFiniteModel().model_dump_json())
    assert loaded["values"][:2] == ["-Infinity", "Infinity"]
    assert loaded["values"][2] == "NaN"
    reloaded = NonFiniteModel.model_validate(loaded)
    assert reloaded.values[0] == -float("inf")
    assert reloaded.values[1] == float("inf")
    assert reloaded.values[2] != reloaded.values[2]


def test_xoptbase_public_io_and_whole_file_compression(tmp_path):
    class Model(XoptBaseModel):
        a: int = 1

    model = Model()
    assert json.loads(model.to_json()) == {"a": 1}
    assert yaml.safe_load(model.yaml()) == {"a": 1}
    assert Model.from_dict({"a": 2}).a == 2
    assert Model.from_yaml(io.StringIO("a: 3\n")).a == 3

    plain = tmp_path / "model.yaml"
    plain.write_text("a: 4\n")
    assert Model.from_file(str(plain)).a == 4
    compressed = tmp_path / "model.yaml.gz"
    compressed.write_bytes(gzip.compress(b"a: 5\n"))
    assert Model.from_file(str(compressed)).a == 5
    with pytest.raises(OSError):
        Model.from_file(str(tmp_path / "missing.yaml"))


def test_xopt_generator_name_and_serialize_as_any():
    class CustomRandomGenerator(RandomGenerator):
        subclass_only: int = 17

    generator = CustomRandomGenerator(vocs=TEST_VOCS_BASE)
    xopt = Xopt(generator=generator, evaluator=Evaluator(function=misc_fn))

    python_dump = xopt.model_dump()
    json_dump = json.loads(xopt.model_dump_json())
    assert python_dump["generator"]["name"] == generator.name
    assert python_dump["generator"]["subclass_only"] == 17
    assert json_dump["generator"]["name"] == generator.name
    assert json_dump["generator"]["subclass_only"] == 17


def test_callable_serialization_warns_when_not_reloadable(monkeypatch):
    import functools
    import warnings

    import xopt.types

    monkeypatch.setattr(xopt.types, "_WARNED_CALLABLE_NAMES", set())

    class FnModel(XoptBaseModel):
        fn: CallableRef

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dump = json.loads(FnModel(fn=misc_fn).model_dump_json())
    assert dump["fn"] == f"{__name__}.misc_fn"

    model = FnModel(fn=functools.partial(misc_fn, y=3))
    with pytest.warns(UserWarning, match="will not reload"):
        model.model_dump_json()
    # warned once per callable per process, not once per dump
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.model_dump_json()

    # lambdas have no importable qualified name
    with pytest.warns(UserWarning, match="will not reload"):
        FnModel(fn=lambda x: x).model_dump_json()


@pytest.mark.parametrize(
    ("bind_args", "bind_kwargs", "build_kwargs", "expected"),
    [
        # tuple defaults degrade to None, None/empty defaults survive
        ((), {}, {"a": 1}, (1, 2, None, None)),
        # bound positional and keyword values override defaults
        ((1,), {"b": 3}, {}, (1, 3, None, None)),
        # build-time kwargs override stored values
        ((1,), {"b": 3}, {"b": 4, "d": "x"}, (1, 4, None, "x")),
    ],
)
def test_validate_and_compose_signature_defaults(
    bind_args, bind_kwargs, build_kwargs, expected
):
    def fn(a, b=2, c=(1, 2), d=None):
        return (a, b, c, d)

    signature = validate_and_compose_signature(fn, *bind_args, **bind_kwargs)
    args, kwargs = signature.build(**build_kwargs)
    assert fn(*args, **kwargs) == expected


def test_validate_and_compose_signature_varargs_and_invalid():
    def fn(*args, x=1):
        return args, x

    signature = validate_and_compose_signature(fn, 1, 2)
    assert signature.model_dump() == {"args": [1, 2], "x": 1}
    # partial positional replacement keeps the remaining stored args
    args, kwargs = signature.build(4)
    assert (args, kwargs) == ([4, 2], {"x": 1})

    def plain(a, b=2):
        return a, b

    with pytest.raises(TypeError, match="too many positional"):
        validate_and_compose_signature(plain, 1, 2, 3)
    with pytest.raises(TypeError, match="unexpected keyword"):
        validate_and_compose_signature(plain, nope=1)


def test_callable_model_reload_and_bind():
    model = CallableModel(callable=misc_fn, kwargs={"y": 7})
    dumped = json.loads(model.model_dump_json())
    assert dumped["callable"] == f"{__name__}.misc_fn"
    reloaded = CallableModel.model_validate(dumped)
    assert reloaded(x=5) == 12
    # positional call args map through the stored kwarg_order
    assert reloaded(3, 4) == 7

    instance = MiscClass(value=10)
    bound = CallableModel(callable=f"{__name__}.MiscClass.misc_method", bind=instance)
    assert bound(x=3) == 13
    with pytest.raises(ValueError, match="Cannot bind"):
        CallableModel(callable=f"{__name__}.MiscClass.misc_method", bind=object())
    with pytest.raises(ValueError, match="must be object or a string"):
        CallableModel(callable=123)


def test_objloader_guards_and_store():
    # loader callable must match the parameterized type
    with pytest.raises(ValueError):
        ObjLoader[MiscClass].model_validate(
            {"loader": {"callable": f"{__name__}.misc_fn"}}
        )

    loader_dump = json.loads(ObjLoader[MiscClass]().model_dump_json())
    assert isinstance(
        ObjLoader[MiscClass].model_validate(loader_dump).load(), MiscClass
    )

    loader = ObjLoader[MiscClass](kwargs={"value": 5})
    assert loader.object is None
    obj = loader.load(store=True)
    assert loader.object is obj and obj.value == 5


def test_normal_executor_reload_reconstructs_executor():
    executor = NormalExecutor[DummyExecutor](loader={"kwargs": {"tag": "loaded"}})
    dumped = json.loads(executor.model_dump_json())
    assert "executor" not in dumped

    reloaded = NormalExecutor[DummyExecutor].model_validate(dumped)
    assert isinstance(reloaded.executor, DummyExecutor)
    assert reloaded.executor.tag == "loaded"
    assert reloaded.submit(misc_fn, 1, 2) == 3
    reloaded.shutdown()
    assert reloaded.executor.was_shutdown

    with pytest.raises(ValueError, match="instance of DummyExecutor"):
        NormalExecutor[DummyExecutor](executor="not an executor")


def test_dump_and_reload_sidecars_from_other_cwd(tmp_path, monkeypatch):
    from xopt.generators.bayesian.upper_confidence_bound import (
        UpperConfidenceBoundGenerator,
    )

    generator = UpperConfidenceBoundGenerator(
        vocs=TEST_VOCS_BASE,
        gp_constructor=StandardModelConstructor(covar_modules={"y1": RBFKernel()}),
    )
    X = Xopt(
        generator=generator,
        evaluator=Evaluator(function=misc_fn),
        serialize_torch=True,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dump_file = run_dir / "xopt.yaml"
    X.dump(str(dump_file))
    assert (run_dir / "covar_modules_y1.pt").is_file()

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    # from_file resolves sidecars relative to the dump file, not the cwd
    reloaded = Xopt.from_file(str(dump_file))
    assert isinstance(reloaded.generator.gp_constructor.covar_modules["y1"], RBFKernel)

    # an explicit base_dir context must work too, surviving the custom
    # __init__ methods in the generator chain
    config = yaml.safe_load(dump_file.read_text())
    reloaded = Xopt.model_validate(config, context={"base_dir": str(run_dir)})
    assert isinstance(reloaded.generator.gp_constructor.covar_modules["y1"], RBFKernel)


def test_retained_model_helpers():
    class Inner(XoptBaseModel):
        x: int = Field(1, description="x field")

    class Outer(XoptBaseModel):
        inner: Inner = Inner()
        fn: Any = Field(misc_fn, description="function")

    descriptions = get_descriptions_defaults(Outer())
    assert descriptions["inner"]["x"][0] == "x field"
    assert descriptions["fn"][0] == "function"


def test_scalar_and_nonfinite_array_round_trips():
    class ArrayModel(XoptBaseModel):
        array: NDArray
        tensor: TorchTensor

    m = ArrayModel(
        array=np.array([np.nan, 1.0, np.inf]),
        tensor=torch.tensor(2.5, dtype=torch.float64),
    )
    dumped = json.loads(m.model_dump_json())
    assert dumped["array"] == ["NaN", 1.0, "Infinity"]
    assert dumped["tensor"] == 2.5

    loaded = ArrayModel.model_validate(dumped)
    assert loaded.array.dtype == np.float64
    assert np.isnan(loaded.array[0]) and loaded.array[1] == 1.0
    assert np.isinf(loaded.array[2])
    assert loaded.tensor.ndim == 0 and float(loaded.tensor) == 2.5

    assert json.loads(loaded.model_dump_json())["array"] == dumped["array"]

    # legacy lowercase non-finite scalars still load, and bare finite scalars
    # round-trip as 0-d values in both codecs
    legacy = ArrayModel.model_validate({"array": "nan", "tensor": "-inf"})
    assert np.isnan(legacy.array) and float(legacy.tensor) == -np.inf
    scalars = ArrayModel.model_validate({"array": 3.5, "tensor": 0.0})
    assert scalars.array.ndim == 0 and float(scalars.array) == 3.5
    assert scalars.tensor.ndim == 0 and float(scalars.tensor) == 0.0
    assert json.loads(scalars.model_dump_json()) == {"array": 3.5, "tensor": 0.0}


def test_xopt_data_b64_round_trip():
    X = Xopt(
        generator=RandomGenerator(vocs=TEST_VOCS_BASE),
        evaluator=Evaluator(function=misc_fn),
    )
    X.add_data(pd.DataFrame({"x1": [0.1, 0.2], "x2": [0.3, 0.4], "y1": [1.0, 2.0]}))
    reloaded = Xopt.from_yaml(X.yaml(df_mode="b64"))
    assert list(reloaded.data.index) == [0, 1]
    assert reloaded.data["x1"].tolist() == [0.1, 0.2]


def test_module_wrap_serializers_respect_exclude(tmp_path):
    constructor = StandardModelConstructor(covar_modules={"y1": RBFKernel()})
    dumped = json.loads(constructor.model_dump_json(exclude={"covar_modules"}))
    assert "covar_modules" not in dumped
    dumped = json.loads(
        constructor.model_dump_json(
            exclude={"covar_modules"},
            context={"module_mode": "file", "file_dir": tmp_path},
        )
    )
    assert "covar_modules" not in dumped
    assert not list(tmp_path.iterdir()), "excluded field must not write sidecars"


def test_custom_noise_prior_round_trips(tmp_path):
    from gpytorch.priors import GammaPrior

    constructor = StandardModelConstructor(custom_noise_prior=GammaPrior(1.0, 100.0))

    # drop (default): key removed, like other module-valued fields
    dumped = json.loads(constructor.model_dump_json())
    assert "custom_noise_prior" not in dumped
    assert StandardModelConstructor.model_validate(dumped).custom_noise_prior is None

    # inline round trip
    dumped = json.loads(constructor.to_json(module_mode="inline"))
    assert dumped["custom_noise_prior"].startswith("b64pt:")
    loaded = StandardModelConstructor.model_validate(dumped)
    assert isinstance(loaded.custom_noise_prior, GammaPrior)

    # file round trip
    dumped = json.loads(constructor.to_json(module_mode="file", file_dir=str(tmp_path)))
    assert dumped["custom_noise_prior"] == "custom_noise_prior.pt"
    assert (tmp_path / "custom_noise_prior.pt").exists()

    # an unset prior stays as an explicit null in every mode
    empty = json.loads(StandardModelConstructor().model_dump_json())
    assert empty["custom_noise_prior"] is None


class _FirstOutputObjective(CustomXoptObjective):
    # module scope so torch pickling can resolve it
    def forward(self, samples, X=None):
        return samples[..., 0]


def test_custom_objective_round_trips():
    from xopt.generators.bayesian.expected_improvement import (
        ExpectedImprovementGenerator,
    )

    gen = ExpectedImprovementGenerator(
        vocs=TEST_VOCS_BASE, custom_objective=_FirstOutputObjective(TEST_VOCS_BASE)
    )
    dumped = json.loads(gen.model_dump_json())
    assert "custom_objective" not in dumped  # dropped by default, like model

    inline = json.loads(gen.to_json(module_mode="inline"))
    assert inline["custom_objective"].startswith("b64pt:")
    loaded = ExpectedImprovementGenerator.model_validate(inline)
    assert isinstance(loaded.custom_objective, CustomXoptObjective)


def test_model_dump_json_mode_uses_fallback():
    class KwargsModel(XoptBaseModel):
        kwargs: dict = {}

    m = KwargsModel(kwargs={"a": np.float32(1.5), "df": pd.DataFrame({"x": [1]})})
    dumped = m.model_dump(mode="json")
    assert dumped["kwargs"]["a"] == 1.5
    assert dumped["kwargs"]["df"] == {"x": {"0": 1}}
