import copy
import inspect
import json
import logging
import os.path
import typing
from concurrent.futures import Future
from types import BuiltinFunctionType, FunctionType, MethodType
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    List,
    Optional,
    TextIO,
    TypeVar,
)

import numpy as np
import pandas as pd
import torch
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    create_model,
    field_validator,
    model_validator,
)
from pydantic_core.core_schema import ValidationInfo

from xopt.types import (
    CallableRef,
    SerializationOptions,
    TypeRef,
    maybe_decompress,
    module_load_base_dir,
    normalize_serialization_context,
    object_from_qualified_name,
    qualified_name,
    resolve_callable,
)

ObjType = TypeVar("ObjType")
logger = logging.getLogger(__name__)


def serialization_fallback(value: Any) -> Any:
    """Return JSON-native representations for values below untyped fields."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, pd.DataFrame):
        return json.loads(value.to_json())
    if isinstance(value, (type, FunctionType, MethodType, BuiltinFunctionType)):
        # deliberately narrow: callable *instances* (nn.Module, partial, ...)
        # fall through to the generic class-name form below
        return qualified_name(value)
    if isinstance(value, Exception):
        return str(value)
    return f"{type(value).__module__}.{type(value).__qualname__}"


class XoptBaseModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid", ser_json_inf_nan="strings"
    )

    @classmethod
    def model_validate(cls, *args, **kwargs):
        # models with custom __init__ methods re-enter validation without the
        # caller's context, so a context-supplied base_dir must also travel via
        # the contextvar to reach nested file-loading codecs
        context = kwargs.get("context")
        base_dir = context.get("base_dir") if isinstance(context, dict) else None
        if base_dir is not None:
            with module_load_base_dir(base_dir):
                return super().model_validate(*args, **kwargs)
        return super().model_validate(*args, **kwargs)

    def model_dump_json(self, *args, **kwargs) -> str:
        kwargs["context"] = normalize_serialization_context(kwargs.get("context"))
        kwargs.setdefault("fallback", serialization_fallback)
        return super().model_dump_json(*args, **kwargs)

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        if kwargs.get("mode") == "json":
            kwargs["context"] = normalize_serialization_context(kwargs.get("context"))
            kwargs.setdefault("fallback", serialization_fallback)
        return super().model_dump(*args, **kwargs)

    def to_json(self, **kwargs) -> str:
        context = kwargs.pop("context", None)
        option_values = {
            name: kwargs.pop(name)
            for name in (
                "array_mode",
                "module_mode",
                "df_mode",
                "compress",
                "level",
                "file_dir",
            )
            if name in kwargs
        }

        serialize_torch = kwargs.pop("serialize_torch", None)
        serialize_inline = kwargs.pop("serialize_inline", None)
        if "module_mode" not in option_values and (
            serialize_torch is not None or serialize_inline is not None
        ):
            option_values["module_mode"] = (
                "inline"
                if serialize_torch and serialize_inline
                else "file"
                if serialize_torch
                else "drop"
            )

        if isinstance(context, dict):
            normalized_context = normalize_serialization_context(context)
            context_options = normalized_context["serialization_options"]
            merged_values = dict(option_values)
            merged_values.update(
                {
                    name: getattr(context_options, name)
                    for name in context_options._explicit
                }
            )
            merged_context = {
                key: value
                for key, value in normalized_context.items()
                if key not in {"serialization_options", *option_values}
            }
            merged_context["serialization_options"] = SerializationOptions(
                **merged_values
            )
        elif isinstance(context, SerializationOptions):
            merged_values = dict(option_values)
            merged_values.update(
                {name: getattr(context, name) for name in context._explicit}
            )
            merged_context = SerializationOptions(**merged_values)
        elif context is None:
            merged_context = option_values
        else:
            raise TypeError("context must be a mapping or SerializationOptions")
        return self.model_dump_json(context=merged_context, **kwargs)

    def json(self, **kwargs: Any) -> str:
        return self.to_json(**kwargs)

    def yaml(self, **kwargs: Any) -> str:
        """serialize first then dump to yaml string"""
        output = json.loads(self.to_json(**kwargs))
        return yaml.dump(output)

    @classmethod
    def from_file(cls, filename: str) -> "XoptBaseModel":
        if not os.path.exists(filename):
            raise OSError(f"file {filename} is not found")

        with open(filename, "rb") as file:
            raw = maybe_decompress(file.read())
        data = yaml.safe_load(raw.decode("utf-8"))
        base_dir = str(os.path.dirname(os.path.abspath(filename)))
        return cls.model_validate(data, context={"base_dir": base_dir})

    @classmethod
    def from_yaml(cls, yaml_obj: str | TextIO) -> "XoptBaseModel":
        return cls.model_validate(yaml.safe_load(yaml_obj))

    @classmethod
    def from_dict(cls, config: dict) -> "XoptBaseModel":
        return cls.model_validate(config)


def get_descriptions_defaults(model: XoptBaseModel):
    """get a dict containing the descriptions of fields inside nested pydantic models"""

    description_dict: dict[str, Any] = {}
    for name, val in model.model_fields.items():
        value = getattr(model, name)
        # Check if the value is a subclass of XoptBaseModel
        if isinstance(value, XoptBaseModel):
            description_dict[name] = get_descriptions_defaults(value)
        else:
            description_dict[name] = [val.description, val.default]

    return description_dict


class CallableModel(XoptBaseModel):
    callable: CallableRef
    signature: SerializeAsAny[BaseModel]

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @model_validator(mode="before")
    def validate_all(cls, values):
        callable = values.pop("callable")

        if not isinstance(
            callable,
            (
                str,
                Callable,
            ),
        ):
            raise ValueError(
                "Callable must be object or a string. Provided %s", type(callable)
            )

        # parse string to callable
        if isinstance(callable, (str,)):
            # for function loading
            if "bind" in values:
                callable = get_callable_from_string(callable, bind=values.pop("bind"))

            else:
                callable = get_callable_from_string(callable)

        values["callable"] = callable

        # for reloading:
        kwargs = {}
        args = []
        if "args" in values:
            args = values.pop("args")

        if "kwargs" in values:
            kwargs = values.pop("kwargs")

        if "signature" in values:
            if "args" in values["signature"]:
                args = values["signature"].pop("args")

            # not needed during reserialization
            if "kwarg_order" in values["signature"]:
                values["signature"].pop("kwarg_order")

            if "kwargs" in values:
                kwargs = values["signature"]["kwargs"]

            else:
                kwargs = values["signature"]

        values["signature"] = validate_and_compose_signature(callable, *args, **kwargs)

        return values

    def __call__(self, *args, **kwargs):
        if kwargs is None:
            kwargs = {}

        fn_args, fn_kwargs = self.signature.build(*args, **kwargs)

        return self.callable(*fn_args, **fn_kwargs)


class ObjLoader(
    XoptBaseModel,
    Generic[ObjType],
):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    object: Optional[ObjType] = None
    loader: Optional[CallableModel] = None
    object_type: Optional[TypeRef] = None

    @model_validator(mode="before")
    def validate_all(cls, values):
        # In v1, could access type_ to get resolved inner type
        # See https://stackoverflow.com/questions/75165745
        # obj_type = cls.__fields__["object"].type_

        # In v2, how to do this is unclear - internals have changed
        # For now, use hacky way with annotations
        annotation = cls.model_fields["object"].annotation
        # inner_types are (ObjType,NoneType)
        inner_types = typing.get_args(annotation)
        obj_type = inner_types[0]

        # adjust for re init from json
        if "loader" not in values:
            loader = CallableModel(callable=obj_type, **values)

        else:
            # if already-initialized callable, do nothing
            if isinstance(values["loader"], (CallableModel,)):
                loader = values["loader"]

            else:
                # validate loader callable is same as obj type
                if values["loader"].get("callable") is not None:
                    # unparameterized callable will handle parsing
                    callable = CallableModel(callable=values["loader"]["callable"])

                    if callable.callable is not obj_type:
                        raise ValueError(
                            "Provided loader of type %s. ObjLoader parameterized for \
                                %s",
                            callable.callable.__name__,
                            obj_type,
                        )

                    # opt for obj type
                    values["loader"].pop("callable")

                # re-init drop callable from loader vals to use new instance
                loader = CallableModel(callable=obj_type, **values["loader"])

        # update the class json encoders. Will only execute on initial type
        # construction
        # if obj_type not in cls.__config__.json_encoders:
        #    cls.__config__.json_encoders[obj_type] = cls.__config__.json_encoders.pop(
        #        ObjType
        #    )
        return {"object_type": obj_type, "loader": loader}

    def load(self, store: bool = False):
        # store object reference on loader
        if store:
            self.object = self.loader()
            return self.object

        # return loaded object w/o storing
        else:
            return self.loader()


# COMMON BASE FOR EXECUTORS
class BaseExecutor(
    XoptBaseModel,
    Generic[ObjType],
):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # executor_type must comply with https://peps.python.org/pep-3148/ standard
    loader: Optional[ObjLoader[ObjType]] = None  # loader of executor type

    # This is a utility field not included in reps. The typing lib has opened
    # issues on access of generic type within class.
    # This tracks for if-necessary future use.
    executor_type: Optional[TypeRef] = Field(None, exclude=True, validate_default=True)
    submit_callable: str = "submit"
    map_callable: str = "map"
    shutdown_callable: str = "shutdown"

    # executor will not be explicitly serialized, but loaded using loader with class
    # and kwargs
    executor: Optional[ObjType] = Field(None, exclude=True)

    @model_validator(mode="before")
    def validate_all(cls, values):
        # TODO: better solution, since type_ is no longer available
        executor_type = typing.get_args(cls.model_fields["executor"].annotation)[0]

        # check if executor provided
        executor = values.get("executor")
        if executor is not None:
            values.pop("executor")

        # VALIDATE SUBMIT CALLABLE AGAINST EXECUTOR TYPE
        if "submit_callable" not in values:
            # use default
            submit_callable = cls.model_fields["submit_callable"].default
        else:
            submit_callable = values.pop("submit_callable")

        try:
            getattr(executor_type, submit_callable)
        except AttributeError:
            raise ValueError(
                "Executor type %s has no submit method %s.",
                executor_type.__name__,
                submit_callable,
            )

        # VALIDATE MAP CALLABLE AGAINST EXECUTOR TYPE
        if not values.get("map_callable"):
            # use default
            map_callable = cls.model_fields["map_callable"].default
        else:
            map_callable = values.pop("map_callable")

        try:
            getattr(executor_type, map_callable)
        except AttributeError:
            raise ValueError(
                "Executor type %s has no map method %s.",
                executor_type.__name__,
                map_callable,
            )

        # VALIDATE SHUTDOWN CALLABLE AGAINST EXECUTOR TYPE
        if not values.get("shutdown_callable"):
            # use default
            shutdown_callable = cls.model_fields["shutdown_callable"].default
        else:
            shutdown_callable = values.pop("shutdown_callable")

        try:
            getattr(executor_type, shutdown_callable)
        except AttributeError:
            raise ValueError(
                "Executor type %s has no shutdown method %s.",
                executor_type.__name__,
                shutdown_callable,
            )

        # Compose loader utility
        if values.get("loader") is not None:
            loader_values = values.get("loader")
            if isinstance(loader_values, ObjLoader):
                loader = loader_values
            else:
                loader = ObjLoader[executor_type](**loader_values)
        else:
            # maintain reference to original object
            loader_values = copy.copy(values)

            # if executor in values, need to remove
            if "executor" in loader_values:
                loader_values.pop("executor")
            loader = ObjLoader[executor_type](**loader_values)

        # update encoders
        # update the class json encoders. Will only execute on initial type construction
        # if executor_type not in cls.__config__.json_encoders:
        #    cls.__config__.json_encoders[
        #        executor_type
        #    ] = cls.__config__.json_encoders.pop(ObjType)

        return {
            "executor_type": executor_type,
            "submit_callable": submit_callable,
            "shutdown_callable": shutdown_callable,
            "map_callable": map_callable,
            "loader": loader,
            "executor": executor,
        }

    def shutdown(self) -> None:
        shutdown_fn = getattr(self.executor, self.shutdown_callable)
        shutdown_fn()


# NormalExecutor with no context handling on submission and executor persistence
class NormalExecutor(
    BaseExecutor[ObjType],
    Generic[ObjType],
):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # TODO: check if validate_default is sufficient
    @field_validator("executor")
    def validate_executor(cls, v, info: ValidationInfo):
        if v is None:
            v = info.data["loader"].load()

        # if not None, validate against executor type
        else:
            if not isinstance(v, (info.data["executor_type"],)):
                raise ValueError(
                    "Provided executor is not instance of %s",
                    info.data["executor_type"].__name__,
                )

        return v

    def submit(self, fn, *args, **kwargs) -> Future:
        submit_fn = getattr(self.executor, self.submit_callable)
        return submit_fn(fn, *args, **kwargs)

    def map(self, fn, *iter: Iterable, **kwargs) -> Iterable[Future]:
        map_fn = getattr(self.executor, self.map_callable)
        return map_fn(fn, *iter, **kwargs)


def get_callable_from_string(callable: str, bind: Any = None) -> Callable:
    """Get callable from its fully qualified name, e.g. ``module.func`` or
    ``module.Class.method``.

    Parameters
    ----------
    callable : str
        Fully qualified name of the callable.
    bind : Any, optional
        Instance to bind to when the name points to a method of the instance's
        class; the bound method is returned.

    Returns
    -------
    Callable
    """
    fn = resolve_callable(callable)

    if bind is None:
        return fn

    owner_name, _, attr_name = callable.rpartition(".")
    try:
        owner = object_from_qualified_name(owner_name) if owner_name else None
    except ValueError:
        owner = None
    if not isinstance(owner, type) or not isinstance(bind, owner):
        raise ValueError(
            f"Cannot bind {callable!r} to instance of {type(bind).__name__}"
        )

    return getattr(bind, attr_name)


class SignatureModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def build(self, *args, **kwargs):
        stored_kwargs = self.model_dump()

        stored_args = []
        if "args" in stored_kwargs:
            stored_args = stored_kwargs.pop("args")

        # adjust for positional
        args = list(args)
        n_pos_only = len(stored_args)
        positional_kwargs = []
        if len(args) < n_pos_only:
            stored_args[: len(args)] = args

        else:
            stored_args = args[:n_pos_only]
            positional_kwargs = args[n_pos_only:]

        stored_kwargs.update(kwargs)

        # exclude empty parameters
        stored_kwargs = {
            key: value
            for key, value in stored_kwargs.items()
            if not value == inspect.Parameter.empty
        }
        if len(positional_kwargs):
            for i, positional_kwarg in enumerate(positional_kwargs):
                stored_kwargs[self.kwarg_order[i]] = positional_kwarg

        return stored_args, stored_kwargs


def validate_and_compose_signature(callable: Callable, *args, **kwargs):
    # try partial bind to validate
    signature = inspect.signature(callable)
    bound_args = signature.bind_partial(*args, **kwargs)

    sig_kw = bound_args.arguments.get("kwargs", {})
    sig_args = bound_args.arguments.get("args", [])

    sig_kwargs = {}
    # Now go parameter by parameter and assemble kwargs
    for i, param in enumerate(signature.parameters.values()):
        if param.kind in [param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY]:
            # if param not bound use default/ compose field rep
            if not sig_kw.get(param.name):
                # create a field representation
                if param.default == param.empty:
                    sig_kwargs[param.name] = param.empty

                else:
                    sig_kwargs[param.name] = param.default

            else:
                sig_kwargs[param.name] = sig_kw.get(param.name)

            # assign via binding
            if param.name in bound_args.arguments:
                sig_kwargs[param.name] = bound_args.arguments[param.name]

    # create pydantic model
    pydantic_fields = {
        "args": (List[Any], Field(list(sig_args))),
        "kwarg_order": (List[Any], Field(list(sig_kwargs.keys()), exclude=True)),
    }
    for key, value in sig_kwargs.items():
        if isinstance(value, (tuple,)):
            pydantic_fields[key] = (tuple, Field(None))

        elif value == inspect.Parameter.empty:
            pydantic_fields[key] = (inspect.Parameter.empty, Field(value))

        else:
            # assigning empty default
            if value is None:
                pydantic_fields[key] = (inspect.Parameter.empty, Field(None))

            else:
                # Pydantic v2 requires type spec on all fields
                # TODO: maybe raise error on non-primitive types
                pydantic_fields[key] = (type(value), value)

    model = create_model(
        f"Kwargs_{callable.__qualname__}", __base__=SignatureModel, **pydantic_fields
    )

    return model()
