"""Typed, context-aware codecs used by Xopt's pydantic models.

Binary prefixes identify the payload type, while compression is inferred from
magic bytes.  Loading a serialized torch module is intentionally restricted to
module-typed fields and uses ``weights_only=False``; configuration files that
contain modules must therefore be treated as trusted input.
"""

from __future__ import annotations

import base64
import gzip
import importlib
import io
import json
import os
import re
import warnings
import zlib
from contextlib import contextmanager
from types import MethodType
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

import numpy as np
import pandas as pd
import torch
import zstandard
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from pydantic_core.core_schema import SerializationInfo, ValidationInfo

MAX_DECOMPRESSED_SIZE = 256 * 1024 * 1024
_GZIP_MAGIC = b"\x1f\x8b"
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_UNSET = object()
_VALIDATION_BASE_DIR: ContextVar[str | os.PathLike[str] | None] = ContextVar(
    "xopt_validation_base_dir", default=None
)


@dataclass(frozen=True, init=False)
class SerializationOptions:
    """Normalized writer options carried through pydantic serialization context."""

    array_mode: Literal["list", "b64"] = "list"
    module_mode: Literal["drop", "file", "inline"] = "drop"
    df_mode: Literal["dict", "b64"] = "dict"
    compress: Literal["gzip", "zstd"] | None = None
    level: int | None = None
    file_dir: Path = field(default_factory=Path.cwd)
    _explicit: frozenset[str] = field(default_factory=frozenset, repr=False)

    def __init__(
        self,
        *,
        array_mode: Literal["list", "b64"] | object = _UNSET,
        module_mode: Literal["drop", "file", "inline"] | object = _UNSET,
        df_mode: Literal["dict", "b64"] | object = _UNSET,
        compress: Literal["gzip", "zstd"] | None | object = _UNSET,
        level: int | None | object = _UNSET,
        file_dir: str | os.PathLike[str] | object = _UNSET,
    ) -> None:
        values = {
            "array_mode": "list" if array_mode is _UNSET else array_mode,
            "module_mode": "drop" if module_mode is _UNSET else module_mode,
            "df_mode": "dict" if df_mode is _UNSET else df_mode,
            "compress": None if compress is _UNSET else compress,
            "level": None if level is _UNSET else level,
            "file_dir": Path.cwd() if file_dir is _UNSET else Path(file_dir),
        }
        explicit = frozenset(
            name
            for name, value in {
                "array_mode": array_mode,
                "module_mode": module_mode,
                "df_mode": df_mode,
                "compress": compress,
                "level": level,
                "file_dir": file_dir,
            }.items()
            if value is not _UNSET
        )

        if values["array_mode"] not in ("list", "b64"):
            raise ValueError("array_mode must be 'list' or 'b64'")
        if values["module_mode"] not in ("drop", "file", "inline"):
            raise ValueError("module_mode must be 'drop', 'file', or 'inline'")
        if values["df_mode"] not in ("dict", "b64"):
            raise ValueError("df_mode must be 'dict' or 'b64'")
        if values["compress"] not in (None, "gzip", "zstd"):
            raise ValueError("compress must be None, 'gzip', or 'zstd'")

        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_explicit", explicit)

    def resolve(self, name: str, annotation_default: Any = _UNSET) -> Any:
        """Resolve context > annotation > library-default precedence."""
        if name in self._explicit or annotation_default is _UNSET:
            return getattr(self, name)
        return annotation_default


_OPTION_NAMES = {
    "array_mode",
    "module_mode",
    "df_mode",
    "compress",
    "level",
    "file_dir",
}


def normalize_serialization_context(context: Any = None) -> dict[str, Any]:
    """Return a context dict containing one normalized options object."""
    if isinstance(context, SerializationOptions):
        return {"serialization_options": context}
    if context is None:
        return {"serialization_options": SerializationOptions()}
    if not isinstance(context, dict):
        raise TypeError(
            "serialization context must be a mapping or SerializationOptions"
        )

    result = dict(context)
    existing = result.get("serialization_options", result.get("options"))
    raw = {key: result[key] for key in _OPTION_NAMES if key in result}
    if existing is not None and not isinstance(existing, SerializationOptions):
        if not isinstance(existing, dict):
            raise TypeError(
                "serialization_options must be a mapping or SerializationOptions"
            )
        raw = {**existing, **raw}
        existing = None

    if existing is None:
        options = SerializationOptions(**raw)
    elif raw:
        base = {name: getattr(existing, name) for name in existing._explicit}
        options = SerializationOptions(**{**base, **raw})
    else:
        options = existing
    result["serialization_options"] = options
    result.pop("options", None)
    return result


def get_serialization_options(context: Any = None) -> SerializationOptions:
    return normalize_serialization_context(context)["serialization_options"]


@contextmanager
def module_load_base_dir(base_dir: str | os.PathLike[str]):
    """Preserve file-relative loading through models with custom ``__init__`` methods."""
    token = _VALIDATION_BASE_DIR.set(base_dir)
    try:
        yield
    finally:
        _VALIDATION_BASE_DIR.reset(token)


def _compress(raw: bytes, algorithm: str | None, level: int | None) -> bytes:
    if algorithm is None:
        return raw
    if algorithm == "gzip":
        return gzip.compress(raw, compresslevel=9 if level is None else level)
    if algorithm == "zstd":
        kwargs = {} if level is None else {"level": level}
        return zstandard.ZstdCompressor(**kwargs).compress(raw)
    raise ValueError(f"unknown compression algorithm: {algorithm}")


def maybe_decompress(raw: bytes, *, max_size: int = MAX_DECOMPRESSED_SIZE) -> bytes:
    """Decompress gzip/zstd data inferred by magic bytes, enforcing a size cap."""
    if raw.startswith(_GZIP_MAGIC):
        decompressor = zlib.decompressobj(wbits=31)
        try:
            result = decompressor.decompress(raw, max_size + 1)
            if len(result) > max_size or decompressor.unconsumed_tail:
                raise ValueError("decompressed payload exceeds the size limit")
            result += decompressor.flush(max_size + 1 - len(result))
        except zlib.error as exc:
            raise ValueError("invalid gzip-compressed payload") from exc
        if len(result) > max_size:
            raise ValueError("decompressed payload exceeds the size limit")
        if not decompressor.eof or decompressor.unused_data:
            raise ValueError("invalid gzip-compressed payload")
        return result

    if raw.startswith(_ZSTD_MAGIC):
        try:
            result = zstandard.ZstdDecompressor().decompress(
                raw, max_output_size=max_size + 1
            )
        except zstandard.ZstdError as exc:
            raise ValueError("invalid zstd-compressed payload") from exc
        if len(result) > max_size:
            raise ValueError("decompressed payload exceeds the size limit")
        return result

    # the cap guards decompression amplification only
    return raw


def _b64encode(prefix: str, raw: bytes) -> str:
    return prefix + base64.b64encode(raw).decode("ascii")


def _b64decode(value: str, prefix: str) -> bytes:
    if not value.startswith(prefix):
        raise ValueError(f"expected a {prefix} payload")
    try:
        return base64.b64decode(value[len(prefix) :], validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError(f"malformed base64 data in {prefix} payload") from exc


def sanitize_sidecar_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _is_nonfinite_token(value: Any) -> bool:
    return isinstance(value, str) and value.lower().lstrip("+-") in (
        "nan",
        "inf",
        "infinity",
    )


def _coerce_nonfinite_tokens(value: Any) -> Any:
    """Convert "NaN"/"Infinity"-style strings (ser_json_inf_nan output, and the
    legacy walker's lowercase forms) back to floats inside nested lists."""
    if isinstance(value, list):
        return [_coerce_nonfinite_tokens(item) for item in value]
    if _is_nonfinite_token(value):
        return float(value)
    return value


def sidecar_path(options: SerializationOptions, name: str) -> tuple[Path, str]:
    filename = sanitize_sidecar_name(name)
    directory = Path(options.file_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename, filename


# sidecar files this process has written; replacing one of these is a routine
# checkpoint overwrite, replacing anything else clobbers a foreign file
_WRITTEN_SIDECARS: set[Path] = set()


def save_module_sidecar(
    module: torch.nn.Module, options: SerializationOptions, name: str
) -> str:
    """Atomically write a module sidecar file; returns the serialized filename."""
    # TODO: consider prefixing sidecar names with the dump-file stem so dumps
    # sharing a directory (or keys that sanitize identically) cannot collide
    path, filename = sidecar_path(options, name)
    key = path.resolve()
    if path.exists() and key not in _WRITTEN_SIDECARS:
        warnings.warn(
            f"overwriting existing sidecar file {path}", UserWarning, stacklevel=2
        )
    _WRITTEN_SIDECARS.add(key)
    temp_path = path.with_name(path.name + ".tmp")
    torch.save(module, temp_path)
    os.replace(temp_path, path)
    return filename


class NDArrayCodec:
    def __init__(
        self,
        *,
        array_mode: Literal["list", "b64"] | object = _UNSET,
        compress: Literal["gzip", "zstd"] | None | object = _UNSET,
        level: int | None | object = _UNSET,
    ) -> None:
        self.array_mode = array_mode
        self.compress = compress
        self.level = level

    @staticmethod
    def validate(value: Any, _: ValidationInfo) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, list):
            return np.asarray(_coerce_nonfinite_tokens(value))
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            # 0-d arrays serialize as bare scalars via tolist()
            return np.asarray(float(value))
        if isinstance(value, str) and value.startswith("b64np:"):
            raw = maybe_decompress(_b64decode(value, "b64np:"))
            try:
                result = np.load(io.BytesIO(raw), allow_pickle=False)
            except Exception as exc:
                raise ValueError("invalid b64np: NumPy payload") from exc
            if not isinstance(result, np.ndarray):
                raise ValueError("b64np: payload did not contain an ndarray")
            return result
        if _is_nonfinite_token(value):
            # non-finite scalars are dumped as strings (ser_json_inf_nan);
            # legacy dumps also encoded 0-d nan/inf arrays this way
            return np.asarray(float(value))
        raise ValueError("expected a NumPy array, list, scalar, or b64np: string")

    def serialize(self, value: np.ndarray, info: SerializationInfo) -> Any:
        options = get_serialization_options(info.context)
        mode = options.resolve("array_mode", self.array_mode)
        if mode == "list":
            return value.tolist()
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        compress = options.resolve("compress", self.compress)
        level = options.resolve("level", self.level)
        return _b64encode("b64np:", _compress(buffer.getvalue(), compress, level))

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.with_info_plain_validator_function(
            self.validate,
            json_schema_input_schema=core_schema.union_schema(
                [core_schema.list_schema(), core_schema.str_schema()]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                self.serialize, info_arg=True, when_used="json"
            ),
        )


class TorchTensorCodec:
    def __init__(
        self,
        *,
        array_mode: Literal["list", "b64"] | object = _UNSET,
        compress: Literal["gzip", "zstd"] | None | object = _UNSET,
        level: int | None | object = _UNSET,
    ) -> None:
        self.array_mode = array_mode
        self.compress = compress
        self.level = level

    @staticmethod
    def validate(value: Any, _: ValidationInfo) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, list):
            return torch.tensor(_coerce_nonfinite_tokens(value))
        if isinstance(value, (int, float)):
            # 0-d tensors serialize as bare scalars via tolist()
            return torch.tensor(value)
        if _is_nonfinite_token(value):
            return torch.tensor(float(value))
        if isinstance(value, str) and value.startswith("b64pt:"):
            raw = maybe_decompress(_b64decode(value, "b64pt:"))
            try:
                result = torch.load(io.BytesIO(raw), weights_only=True)
            except Exception as exc:
                raise ValueError("invalid b64pt: tensor payload") from exc
            if not isinstance(result, torch.Tensor):
                raise ValueError("b64pt: tensor payload did not contain a tensor")
            return result
        raise ValueError("expected a torch tensor, list, or b64pt: string")

    def serialize(self, value: torch.Tensor, info: SerializationInfo) -> Any:
        options = get_serialization_options(info.context)
        mode = options.resolve("array_mode", self.array_mode)
        if mode == "list":
            return value.detach().cpu().tolist()
        buffer = io.BytesIO()
        torch.save(value.detach().cpu(), buffer)
        compress = options.resolve("compress", self.compress)
        level = options.resolve("level", self.level)
        return _b64encode("b64pt:", _compress(buffer.getvalue(), compress, level))

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.with_info_plain_validator_function(
            self.validate,
            json_schema_input_schema=core_schema.union_schema(
                [core_schema.list_schema(), core_schema.str_schema()]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                self.serialize, info_arg=True, when_used="json"
            ),
        )


def encode_torch_module(
    module: torch.nn.Module,
    *,
    compress: Literal["gzip", "zstd"] | None = None,
    level: int | None = None,
) -> str:
    """Encode a trusted torch module as a typed inline payload."""
    buffer = io.BytesIO()
    torch.save(module, buffer, pickle_protocol=5)
    return _b64encode("b64pt:", _compress(buffer.getvalue(), compress, level))


def decode_torch_module(value: str, *, base_dir: str | os.PathLike[str] | None = None):
    """Decode a trusted inline/path torch module (uses ``weights_only=False``)."""
    if value.startswith("base64:"):
        # legacy inline prefix written by pre-3.3 releases; the payload is the
        # same base64 of torch.save as b64pt:
        value = "b64pt:" + value[len("base64:") :]
    if value.startswith("b64pt:"):
        raw = maybe_decompress(_b64decode(value, "b64pt:"))
        try:
            return torch.load(io.BytesIO(raw), weights_only=False)
        except Exception as exc:
            raise ValueError("invalid b64pt: torch module payload") from exc

    if base_dir is None:
        base_dir = _VALIDATION_BASE_DIR.get()
    candidates = [Path(value)]
    if base_dir is not None and not Path(value).is_absolute():
        candidates.append(Path(base_dir) / value)
    for candidate in candidates:
        if candidate.exists():
            return torch.load(candidate, weights_only=False)
    raise ValueError(f"cannot load torch module from {value}")


class TorchModuleCodec:
    def __init__(
        self,
        *,
        compress: Literal["gzip", "zstd"] | None | object = _UNSET,
        level: int | None | object = _UNSET,
    ) -> None:
        self.compress = compress
        self.level = level

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        def validate(value: Any, info: ValidationInfo) -> Any:
            if isinstance(value, str):
                context = info.context if isinstance(info.context, dict) else {}
                value = decode_torch_module(value, base_dir=context.get("base_dir"))
            if not isinstance(value, source_type):
                raise ValueError(f"expected a {source_type.__qualname__} torch module")
            return value

        def serialize(value: torch.nn.Module, info: SerializationInfo) -> Any:
            options = get_serialization_options(info.context)
            if options.module_mode != "inline":
                # Owning-model wrap serializers remove or replace this placeholder.
                return None
            compress = options.resolve("compress", self.compress)
            level = options.resolve("level", self.level)
            return encode_torch_module(value, compress=compress, level=level)

        return core_schema.with_info_plain_validator_function(
            validate,
            json_schema_input_schema=core_schema.str_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize, info_arg=True, when_used="json"
            ),
        )


class TorchDTypeCodec:
    @staticmethod
    def validate(value: Any, _: ValidationInfo) -> torch.dtype:
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str) and value.startswith("torch."):
            result = getattr(torch, value.removeprefix("torch."), None)
            if isinstance(result, torch.dtype):
                return result
        raise ValueError(f"invalid torch dtype: {value!r}")

    @staticmethod
    def serialize(value: torch.dtype, _: SerializationInfo) -> str:
        return str(value)

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.with_info_plain_validator_function(
            self.validate,
            json_schema_input_schema=core_schema.str_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                self.serialize, info_arg=True, when_used="json"
            ),
        )


class DataFrameCodec:
    def __init__(
        self,
        *,
        df_mode: Literal["dict", "b64"] | object = _UNSET,
        compress: Literal["gzip", "zstd"] | None | object = _UNSET,
        level: int | None | object = _UNSET,
    ) -> None:
        self.df_mode = df_mode
        self.compress = compress
        self.level = level

    @staticmethod
    def _restore_index(df: pd.DataFrame) -> pd.DataFrame:
        # JSON object keys are always strings; recover the integer index a
        # round trip started with so reloads match the original frame
        # (pandas >= 3 uses the dedicated "str" dtype, older versions "object")
        if pd.api.types.is_string_dtype(df.index) or df.index.dtype == object:
            try:
                df.index = df.index.astype(np.int64)
            except (TypeError, ValueError):
                pass
        return df

    @staticmethod
    def validate(value: Any, _: ValidationInfo) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            return value
        if isinstance(value, dict):
            return DataFrameCodec._restore_index(pd.DataFrame(value))
        if isinstance(value, str) and value.startswith("b64df:"):
            raw = maybe_decompress(_b64decode(value, "b64df:"))
            try:
                # conversion heuristics disabled so decoding matches the plain
                # dict mode (no date sniffing)
                result = pd.read_json(
                    io.StringIO(raw.decode("utf-8")),
                    orient="columns",
                    convert_axes=False,
                    convert_dates=False,
                )
            except Exception as exc:
                raise ValueError("invalid b64df: DataFrame payload") from exc
            return DataFrameCodec._restore_index(result)
        raise ValueError("expected a DataFrame, dict, or b64df: string")

    def serialize(self, value: pd.DataFrame, info: SerializationInfo) -> Any:
        options = get_serialization_options(info.context)
        mode = options.resolve("df_mode", self.df_mode)
        text = value.to_json()
        if mode == "dict":
            return json.loads(text)
        compress = options.resolve("compress", self.compress)
        level = options.resolve("level", self.level)
        return _b64encode("b64df:", _compress(text.encode(), compress, level))

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.with_info_plain_validator_function(
            self.validate,
            json_schema_input_schema=core_schema.union_schema(
                [core_schema.dict_schema(), core_schema.str_schema()]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                self.serialize, info_arg=True, when_used="json"
            ),
        )


def qualified_name(value: Callable[..., Any] | type) -> str:
    qualname = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    module = getattr(value, "__module__", None)
    if qualname is None or module is None:
        # callable instances (functools.partial, nn.Module, ...) have no
        # qualname of their own
        cls = type(value)
        return f"{cls.__module__}.{cls.__qualname__}"
    return f"{module}.{qualname}"


def object_from_qualified_name(value: str) -> Any:
    parts = value.split(".")
    for split_at in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split_at])
        try:
            result = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            # only fall back to a shorter prefix when the prefix itself is not
            # a module; a missing transitive dependency must propagate, or an
            # unrelated attribute could shadow the intended submodule
            missing = error.name or ""
            if missing == module_name or module_name.startswith(missing + "."):
                continue
            raise
        for attribute in parts[split_at:]:
            try:
                result = getattr(result, attribute)
            except AttributeError as error:
                raise ValueError(f"cannot import object from {value!r}") from error
        return result
    raise ValueError(f"cannot import object from {value!r}")


def resolve_callable(value: Any) -> Callable[..., Any]:
    """Return ``value`` if it is already callable, otherwise import it from its
    fully qualified name (e.g. ``"math.sqrt"`` or ``"module.Class.method"``)."""
    if callable(value):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{value!r} must be a callable or a qualified-name string")
    result = object_from_qualified_name(value)
    if not callable(result):
        raise ValueError(f"{value!r} does not name a callable")
    return result


def _serialized_name_round_trips(value: Any, name: str) -> bool:
    """True if resolving ``name`` recovers ``value`` (lossless serialization)."""
    try:
        resolved = object_from_qualified_name(name)
    except (ValueError, ModuleNotFoundError):
        return False
    if resolved is value:
        return True
    # bound-method objects are created anew on each attribute access, so
    # identity fails even for a lossless round trip; compare the parts
    if isinstance(value, MethodType) and isinstance(resolved, MethodType):
        return (
            resolved.__func__ is value.__func__ and resolved.__self__ is value.__self__
        )
    return False


_WARNED_CALLABLE_NAMES: set[str] = set()


def _warn_callable_once(name: str, message: str) -> None:
    # periodic dump_file checkpointing serializes after every batch; warn once
    # per callable per process instead of once per dump
    if name not in _WARNED_CALLABLE_NAMES:
        _WARNED_CALLABLE_NAMES.add(name)
        warnings.warn(message, UserWarning)


class CallableCodec:
    @staticmethod
    def validate(value: Any, _: ValidationInfo) -> Callable[..., Any]:
        return resolve_callable(value)

    @staticmethod
    def serialize(value: Callable[..., Any], _: SerializationInfo) -> str:
        name = qualified_name(value)
        # TODO: change these warnings to raise ValueError in a future release
        if not _serialized_name_round_trips(value, name):
            _warn_callable_once(
                name,
                f"serialized callable {name!r} will not reload as the same object "
                "(lambda, functools.partial, bound method, or closure); pass a "
                "module-level callable instead",
            )
        elif name.split(".", 1)[0] == "__main__":
            _warn_callable_once(
                name,
                f"serialized callable {name!r} is defined in __main__ and will "
                "only reload from a process that defines it (e.g. re-running "
                "the same script); define it in an importable module for "
                "portable dumps",
            )
        return name

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.with_info_plain_validator_function(
            self.validate,
            json_schema_input_schema=core_schema.str_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                self.serialize, info_arg=True, when_used="json"
            ),
        )


class TypeCodec:
    @staticmethod
    def validate(value: Any, _: ValidationInfo) -> type:
        if isinstance(value, str):
            value = object_from_qualified_name(value)
        if not isinstance(value, type):
            raise ValueError("expected a type or qualified-name string")
        return value

    @staticmethod
    def serialize(value: type, _: SerializationInfo) -> str:
        return qualified_name(value)

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.with_info_plain_validator_function(
            self.validate,
            json_schema_input_schema=core_schema.str_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                self.serialize, info_arg=True, when_used="json"
            ),
        )


NDArray = Annotated[np.ndarray, NDArrayCodec()]
TorchTensor = Annotated[torch.Tensor, TorchTensorCodec()]
TorchDType = Annotated[torch.dtype, TorchDTypeCodec()]
XDataFrame = Annotated[pd.DataFrame, DataFrameCodec()]
CallableRef = Annotated[Callable[..., Any], CallableCodec()]
TypeRef = Annotated[type, TypeCodec()]
