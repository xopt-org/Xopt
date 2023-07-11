import copy
import inspect
import json
import logging
import os.path
import typing
from concurrent.futures import Future
from importlib import import_module
from types import FunctionType, MethodType
from typing import Any, Callable, Generic, Iterable, List, Optional, TypeVar

import numpy as np
import orjson
import pandas as pd
import torch.nn
from pydantic import BaseModel, ConfigDict, create_model, Field, field_serializer, field_validator, \
    model_serializer, model_validator, validator
from pydantic_core.core_schema import FieldValidationInfo

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


def recursive_serialize(v, base_key="", serialize_torch=False):
    for key in list(v):
        if isinstance(v[key], dict):
            v[key] = recursive_serialize(v[key], key, serialize_torch)
        elif isinstance(v[key], torch.nn.Module):
            if serialize_torch:
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


def recursive_serialize_v2(v, base_key=""):
    #Pydantic v2 will by default serialize submodels as annotated, dropping subclass attributes
    #We don't want that, so need to modify things to SerializeAsAny later

    # This will iterate model fields
    for key, value in v.items():
        if isinstance(value, dict):
            v[key] = recursive_serialize(value, key)
        elif isinstance(value, torch.nn.Module):
            v[key] = process_torch_module(module=value, name="_".join((base_key, key)))
        elif isinstance(value, torch.dtype):
            v[key] = str(value)
        elif isinstance(value, BaseModel):
            recursive_serialize_v2(value, key)
        else:
            for _type, func in JSON_ENCODERS.items():
                if isinstance(value, _type):
                    v[key] = func(value)

        # check to make sure object has been serialized,
        # if not use a generic serializer
        try:
            json.dumps(v[key])
        except (TypeError, OverflowError):
            v[key] = f"{v[key].__module__}.{v[key].__class__.__qualname__}"

    return v


DECODERS = {"torch.float32": torch.float32, "torch.float64": torch.float64}


def recursive_deserialize(v: dict):
    """deserialize strings from xopt outputs"""
    for key, value in v.items():
        # process dicts
        if isinstance(value, dict):
            v[key] = recursive_deserialize(value)

        elif isinstance(value, str):
            if value in DECODERS:
                v[key] = DECODERS[value]

    return v

def recursive_deserialize_v2(v: dict):
    """deserialize strings from xopt outputs"""
    for key, value in v.items():
        # process dicts
        if isinstance(value, dict):
            v[key] = recursive_deserialize(value)

        elif isinstance(value, str):
            if value in DECODERS:
                v[key] = DECODERS[value]

    return v


# define custom json_dumps using orjson
def orjson_dumps(v, *, default, base_key="", serialize_torch=False):
    v = recursive_serialize(v, base_key=base_key, serialize_torch=serialize_torch)
    # orjson.dumps returns bytes, to match standard json.dumps we need to decode
    return orjson.dumps(v, default=default).decode()


def orjson_dumps_v2(v: BaseModel, base_key=""):
    v = recursive_serialize_v2(v, base_key=base_key)
    return orjson.dumps(v).decode()


def orjson_loads(v, default=None):
    v = orjson.loads(v)
    v = recursive_deserialize(v)
    return v


def orjson_loads_v2(v, default=None):
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


class XoptBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    @field_validator("*", mode='before')
    def validate_files(cls, value, info: FieldValidationInfo):
        if isinstance(value, str):
            if os.path.exists(value):
                extension = value.split(".")[-1]
                if extension == "pt":
                    value = torch.load(value)

        return value

    @model_serializer(mode='plain', when_used='json', return_type='str')
    def serialize(self):
        return orjson_dumps_v2(self)

    #TODO: implement json load parsing on main object (json_loads is gone)

    # @model_validator(mode='before')
    # def validate_files(cls, values):
    #     if isinstance(values, BaseModel):
    #         raise ValueError(f'This pydantic mode is poorly documented?')
    #     for field_name in values.keys():
    #         value = values[field_name]
    #         if isinstance(value, str):
    #             if os.path.exists(value):
    #                 extension = value.split(".")[-1]
    #                 if extension == "pt":
    #                     values[field_name] = torch.load(value)
    #     return values

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

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    @model_serializer(mode='plain', when_used='json', return_type='str')
    def serialize(self):
        return orjson_dumps_v2(self)

    @model_validator(mode='before')
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

    @model_serializer(mode='plain', when_used='json', return_type='str')
    def serialize(self):
        return orjson_dumps_v2(self)

    @model_validator(mode='before')
    def validate_all(cls, values):
        # inspect class init signature
        #obj_type = cls.__fields__["object"].type_
        # In v1, could access type_ to get resolved inner type
        # See https://stackoverflow.com/questions/75165745/cannot-determine-if-type-of-field-in-a-pydantic-model-is-of-type-list

        # In v2, how to do this is unclear - internals have changed
        # TODO: redo this hack
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

    @model_validator(mode='after')
    def validate_print(cls, values):
        print(values)
        return values

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

    @model_validator(mode='before')
    def validate_all(cls, values):
        print('model validator before: ', values)
        annotation = cls.model_fields["object"].annotation
        inner_types = typing.get_args(annotation)
        obj_type = inner_types[0]
        print(f'{obj_type=}')
        return {"object_type": obj_type}

    @model_validator(mode='after')
    def validate_print(cls, values):
        print('model validator after: ', values)
        return values

    @field_serializer('object_type', when_used='json')
    def serialize_object_type(self, x):
        print('object_type serializer', x)
        if x is None:
            return x
        return f"{x.__module__}.{x.__name__}"


# COMMON BASE FOR EXECUTORS
class BaseExecutor(
        BaseModel,
        Generic[ObjType],

):
    model_config = {'arbitrary_types_allowed': True,
                    # Needed to avoid: https://github.com/samuelcolvin/pydantic/discussions/4099
                    # TODO: check in v2
                    'copy_on_model_validation': 'none',
                    }

    # executor_type must comply with https://peps.python.org/pep-3148/ standard
    loader: Optional[ObjLoader[ObjType]] = None  # loader of executor type

    # This is a utility field not included in reps. The typing lib has opened
    # issues on access of generic type within class.
    # This tracks for if-necessary future use.
    executor_type: type = Field(None, exclude=True)
    submit_callable: str = "submit"
    map_callable: str = "map"
    shutdown_callable: str = "shutdown"

    # executor will not be explicitely serialized, but loaded using loader with class
    # and kwargs
    executor: Optional[ObjType] = None

    @model_validator(mode='before')
    def validate_all(cls, values):
        executor_type = cls.__fields__[
            "executor"
        ].type_  # introspect fields to get type

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
    model_config = {'arbitrary_types_allowed': True,
                    #'json_dumps': orjson_dumps,
                    }

    @model_serializer(mode='plain', when_used='json', return_type='str')
    def serialize(self):
        return orjson_dumps_v2(self)

    # TODO: verify if new style works same as 'always'
    @validator("executor", always=True)
    def validate_executor(cls, v, values):
        if v is None:
            v = values["loader"].load()

        # if not None, validate against executor type
        else:
            if not isinstance(v, (values["executor_type"],)):
                raise ValueError(
                        "Provided executor is not instance of %s",
                        values["executor_type"].__name__,
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

    Args:
        callable: String representation of callable abiding convention
             __module__:callable
        bind: Class to bind as self

    Returns:
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
    class Config:
        arbitrary_types_allowed = True

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
                pydantic_fields[key] = value

    model = create_model(
            f"Kwargs_{callable.__qualname__}", __base__=SignatureModel, **pydantic_fields
    )

    return model()
