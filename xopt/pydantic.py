import copy
import inspect
import io
import json
import logging
import os.path
import typing
from concurrent.futures import Future
from functools import partial
from importlib import import_module
from types import FunctionType, MethodType
from typing import Any, Callable, Generic, Iterable, List, Optional, TextIO, TypeVar

import numpy as np
import orjson
import pandas as pd
import torch.nn
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    create_model,
    Field,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.v1.json import custom_pydantic_encoder
from pydantic_core.core_schema import SerializationInfo, ValidationInfo

ObjType = TypeVar("ObjType")
logger = logging.getLogger(__name__)

JSON_ENCODERS = {
    # function/method type distinguished for class members
    # and not recognized as callables
    FunctionType: lambda x: f"{x.__module__}.{x.__qualname__}",
    MethodType: lambda x: f"{x.__module__}.{x.__qualname__}",
    Callable: lambda x: f"{x.__module__}.{x.__qualname__}",
    type: lambda x: f"{x.__module__}.{x.__name__}",
    # for encoding instances of the ObjType}
    # ObjType: lambda x: f"{x.__module__}.{x.__class__.__qualname__}",
    np.ndarray: lambda x: x.tolist(),
    np.int64: lambda x: int(x),
    np.float64: lambda x: float(x),
    # torch.nn.Module: lambda x: process_torch_module(x),
    # torch.Tensor: lambda x: x.detach().cpu().numpy().tolist(),
}


# The problem with v2 serialization is that model_serialize_json() does not accept kwargs
# meaning whichever model method is decorated with @model_serializer cant adjust for 'base_key'
# and other similar options - it renders native whole v2 scheme quite useless. We can still try
# to use @field_serializer, but there is a lack of documentation on how to call these
# handlers from a custom function.
#
# So, we implement two serialization options for now.
# First is the native one, with no customization, under serialize_json() method. It needs to
# invoked as xopt.model_dump_json(), the standard pydantic v2 syntax.
# Second method bypasses pydantic completely. It is invoked via 'xopt.json()'
# or '<any xopt model>.to_json()'

# Pydantic v2 will by default serialize submodels as annotated types, dropping subclass attributes


def recursive_serialize(
    v, base_key="", serialize_torch=False, serialize_inline: bool = False
) -> dict:
    for key in list(v):
        if isinstance(v[key], dict):
            v[key] = recursive_serialize(v[key], key, serialize_torch, serialize_inline)
        elif isinstance(v[key], torch.nn.Module):
            if serialize_torch:
                if serialize_inline:
                    v[key] = "base64:" + encode_torch_module(v[key])
                else:
                    v[key] = process_torch_module(
                        module=v[key], name="_".join((base_key, key))
                    )
            else:
                del v[key]
        elif isinstance(v[key], torch.dtype):
            v[key] = str(v[key])
        elif isinstance(v[key], pd.DataFrame):
            v[key] = json.loads(v[key].to_json())
        else:
            for _type, func in JSON_ENCODERS.items():
                if isinstance(v[key], _type):
                    v[key] = func(v[key])

        # check to make sure object has been serialized,
        # if not use a generic serializer
        try:
            # handle case when key is (not) deleted
            if key in v:
                json.dumps(v[key])
        except (TypeError, OverflowError):
            v[key] = f"{v[key].__module__}.{v[key].__class__.__qualname__}"

    return v


DECODERS = {"torch.float32": torch.float32, "torch.float64": torch.float64}


def recursive_deserialize(v: dict) -> dict:
    """deserialize strings from xopt outputs"""
    for key, value in v.items():
        # process dicts
        if isinstance(value, dict):
            v[key] = recursive_deserialize(value)

        elif isinstance(value, str):
            if value in DECODERS:
                v[key] = DECODERS[value]

    return v


def orjson_dumps(
    v: BaseModel, *, base_key="", serialize_torch=False, serialize_inline=False
) -> str:
    # TODO: move away from borrowing pydantic v1 encoder preset
    json_encoder = partial(custom_pydantic_encoder, JSON_ENCODERS)
    return orjson_dumps_custom(
        v,
        default=json_encoder,
        base_key=base_key,
        serialize_torch=serialize_torch,
        serialize_inline=serialize_inline,
    )


def orjson_dumps_custom(v: BaseModel, *, default, base_key="", **kwargs) -> str:
    v = recursive_serialize(v.model_dump(), base_key=base_key, **kwargs)
    return orjson.dumps(v, default=default).decode()


def orjson_dumps_except_root(v: BaseModel, *, base_key="", **kwargs) -> dict:
    """Same as above but start at fields of root model, instead of model itself"""
    dump = v.model_dump()
    encoded_dump = recursive_serialize(dump, base_key=base_key, **kwargs)
    return encoded_dump


def orjson_loads(v, default=None) -> dict:
    v = orjson.loads(v)
    v = recursive_deserialize(v)
    return v


def process_torch_module(module, name):
    """save module to file based on module name and return file path to json"""
    # module_name = "".join(random.choices(string.ascii_uppercase + string.digits,
    #                                     k=7)) + ".pt"
    module_name = f"{name}.pt"
    torch.save(module, module_name)
    return module_name


def encode_torch_module(module):
    import base64
    import gzip

    buffer = io.BytesIO()
    # 5 supported since 3.8
    torch.save(module, buffer, pickle_protocol=5)
    module_bytes = buffer.getbuffer().tobytes()
    cb = gzip.compress(module_bytes, compresslevel=9)
    encoded_bytes = base64.standard_b64encode(cb)
    return encoded_bytes.decode("ascii")


def decode_torch_module(modulestr: str):
    import base64
    import gzip

    assert modulestr.startswith("base64:")
    base64str = modulestr.split("base64:", 1)[1]
    decoded = base64.standard_b64decode(base64str)
    decompressed = gzip.decompress(decoded)
    bytestream = io.BytesIO(decompressed)
    module = torch.load(bytestream)
    return module


class XoptBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @field_validator("*", mode="before")
    def validate_files(cls, value, info: ValidationInfo):
        if isinstance(value, str):
            if os.path.exists(value):
                extension = value.split(".")[-1]
                if extension == "pt":
                    value = torch.load(value)

        return value

    # Note that this function still returns a dict, NOT a string. Pydantic will handle
    # final serialization of basic types in Rust.
    @model_serializer(mode="plain", when_used="json")
    def serialize_json(self, sinfo: SerializationInfo) -> dict:
        return orjson_dumps_except_root(self)

    def to_json(self, **kwargs) -> str:
        return orjson_dumps(self, **kwargs)

    def json(self, **kwargs):
        return self.to_json(**kwargs)

    def yaml(self, **kwargs):
        """serialize first then dump to yaml string"""
        output = json.loads(
            self.to_json(
                **kwargs,
            )
        )
        return yaml.dump(output)

    @classmethod
    def from_file(cls, filename: str):
        if not os.path.exists(filename):
            raise OSError(f"file {filename} is not found")

        with open(filename, "r") as file:
            return cls.from_yaml(file)

    @classmethod
    def from_yaml(cls, yaml_obj: [str, TextIO]):
        return cls.model_validate(remove_none_values(yaml.safe_load(yaml_obj)))

    @classmethod
    def from_dict(cls, config: dict):
        return cls.model_validate(remove_none_values(config))


def remove_none_values(d):
    if isinstance(d, dict):
        # Create a copy of the dictionary to avoid modifying the original while iterating
        d = {k: remove_none_values(v) for k, v in d.items() if v is not None}
    elif isinstance(d, list):
        # If it's a list, recursively process each item in the list
        d = [remove_none_values(item) for item in d]
    return d


def get_descriptions_defaults(model: XoptBaseModel):
    """get a dict containing the descriptions of fields inside nested pydantic models"""

    description_dict = {}
    for name, val in model.model_fields.items():
        try:
            if issubclass(getattr(model, name), XoptBaseModel):
                description_dict[name] = get_descriptions_defaults(getattr(model, name))
            else:
                description_dict[name] = [
                    val.description,
                    val.default,
                ]

        except TypeError:
            # if the val is an object or callable type
            description_dict[name] = val.description

    return description_dict


class CallableModel(BaseModel):
    callable: Callable
    signature: BaseModel

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @model_serializer(mode="plain", when_used="json", return_type="str")
    def serialize(self):
        return orjson_dumps(self)

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
    BaseModel,
    Generic[ObjType],
):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    object: Optional[ObjType] = None
    loader: CallableModel = None
    object_type: Optional[type] = None

    @model_serializer(mode="plain", when_used="json", return_type="str")
    def serialize_json(self) -> str:
        return orjson_dumps(self)

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
            self.object = self.loader.call()
            return self.object

        # return loaded object w/o storing
        else:
            return self.loader()


# For testing
class ObjLoaderMinimal(
    BaseModel,
    Generic[ObjType],
):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    object: Optional[ObjType] = None
    object_type: Optional[type] = None

    @model_validator(mode="before")
    def validate_all(cls, values):
        print("model validator before: ", values)
        annotation = cls.model_fields["object"].annotation
        inner_types = typing.get_args(annotation)
        obj_type = inner_types[0]
        print(f"{obj_type=}")
        return {"object_type": obj_type}

    @model_validator(mode="after")
    def validate_print(cls, values):
        print("model validator after: ", values)
        return values

    @field_serializer("object_type", when_used="json")
    def serialize_object_type(self, x):
        print("object_type serializer", x)
        if x is None:
            return x
        return f"{x.__module__}.{x.__name__}"


# COMMON BASE FOR EXECUTORS
class BaseExecutor(
    BaseModel,
    Generic[ObjType],
):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # executor_type must comply with https://peps.python.org/pep-3148/ standard
    loader: Optional[ObjLoader[ObjType]] = None  # loader of executor type

    # This is a utility field not included in reps. The typing lib has opened
    # issues on access of generic type within class.
    # This tracks for if-necessary future use.
    executor_type: Optional[type] = Field(None, exclude=True, validate_default=True)
    submit_callable: str = "submit"
    map_callable: str = "map"
    shutdown_callable: str = "shutdown"

    # executor will not be explicitly serialized, but loaded using loader with class
    # and kwargs
    executor: Optional[ObjType] = None

    @model_serializer(mode="plain", when_used="json", return_type="str")
    def serialize_json(self) -> str:
        return orjson_dumps(self)

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
    """Get callable from a string. In the case that the callable points to a bound method,
    the function returns a callable taking the bind instance as the first arg.

        Parameters
        ----------
        callable: String representation of callable abiding convention
             __module__:callable
        bind: Class to bind as self

    Returns
    -------
        Callable
    """
    callable_split = callable.rsplit(".", 1)

    if len(callable_split) != 2:
        raise ValueError(f"Improperly formatted callable string: {callable_split}")

    module_name, callable_name = callable_split

    try:
        module = import_module(module_name)

    except ModuleNotFoundError:
        try:
            module_split = module_name.rsplit(".", 1)

            if len(module_split) != 2:
                raise ValueError(f"Unable to access: {callable}")

            module_name, class_name = module_split

            module = import_module(module_name)
            callable_name = f"{class_name}.{callable_name}"

        except ModuleNotFoundError as err:
            logger.error("Unable to import module %s", module_name)
            raise err

        except ValueError as err:
            logger.error(err)
            raise err

    # construct partial in case of bound method
    if "." in callable_name:
        bound_class, callable_name = callable_name.rsplit(".")

        try:
            bound_class = getattr(module, bound_class)
        except Exception as e:
            logger.error("Unable to get %s from %s", bound_class, module_name)
            raise e

        # require right partial for assembly of callable
        # https://funcy.readthedocs.io/en/stable/funcs.html#rpartial
        def rpartial(func, *args):
            return lambda *a: func(*(a + args))

        callable = getattr(bound_class, callable_name)
        params = inspect.signature(callable).parameters

        # check bindings
        is_bound = params.get("self", None) is not None
        if not is_bound and bind is not None:
            raise ValueError("Cannot bind %s to %s.", callable_name, bind)

        # bound, return partial
        if bind is not None:
            if not isinstance(bind, (bound_class,)):
                raise ValueError(
                    "Provided bind %s is not instance of %s",
                    bind,
                    bound_class.__qualname__,
                )

        if is_bound and isinstance(callable, (FunctionType,)) and bind is None:
            callable = rpartial(getattr, callable_name)

        elif is_bound and isinstance(callable, (FunctionType,)) and bind is not None:
            callable = getattr(bind, callable_name)

    else:
        if bind is not None:
            raise ValueError("Cannot bind %s to %s.", callable_name, type(bind))

        try:
            callable = getattr(module, callable_name)
        except Exception as e:
            logger.error("Unable to get %s from %s", callable_name, module_name)
            raise e

    return callable


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
            stored_args[:n_pos_only] = args

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
