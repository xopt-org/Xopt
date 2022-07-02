import copy
import inspect
import logging
from concurrent.futures import Future
from importlib import import_module
from types import FunctionType, MethodType
from typing import Any, Callable, Generic, Iterable, List, Optional, TypeVar

import numpy as np
from pydantic import BaseModel, create_model, Extra, Field, root_validator, validator
from pydantic.generics import GenericModel

ObjType = TypeVar("ObjType")
logger = logging.getLogger(__name__)


class XoptBaseModel(BaseModel):
    class Config:
        extra = "forbid"
        json_encoders = {
            np.ndarray: lambda x: x.tolist(),
            np.int64: lambda x: int(x),
            np.float64: lambda x: float(x),
        }


JSON_ENCODERS = {
    # function/method type distinguished for class members
    # and not recognized as callables
    FunctionType: lambda x: f"{x.__module__}.{x.__qualname__}",
    MethodType: lambda x: f"{x.__module__}.{x.__qualname__}",
    Callable: lambda x: f"{x.__module__}.{type(x).__qualname__}",
    type: lambda x: f"{x.__module__}.{x.__name__}",
    # for encoding instances of the ObjType}
    ObjType: lambda x: f"{x.__module__}.{x.__class__.__qualname__}",
}


class CallableModel(BaseModel):
    callable: Callable
    signature: BaseModel

    class Config:
        arbitrary_types_allowed = True
        json_encoders = JSON_ENCODERS
        extra = Extra.forbid

    @root_validator(pre=True)
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
    GenericModel,
    Generic[ObjType],
    arbitrary_types_allowed=True,
    json_encoders=JSON_ENCODERS,
):
    object: Optional[ObjType]
    loader: CallableModel = None
    object_type: Optional[type]

    @root_validator(pre=True)
    def validate_all(cls, values):
        # inspect class init signature
        obj_type = cls.__fields__["object"].type_

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

        # update the class json encoders. Will only execute on initial type construction
        if obj_type not in cls.__config__.json_encoders:
            cls.__config__.json_encoders[obj_type] = cls.__config__.json_encoders.pop(
                ObjType
            )
        return {"object_type": obj_type, "loader": loader}

    def load(self, store: bool = False):
        # store object reference on loader
        if store:
            self.object = self.loader.call()
            return self.object

        # return loaded object w/o storing
        else:
            return self.loader()


# COMMON BASE FOR EXECUTORS
class BaseExecutor(
    GenericModel,
    Generic[ObjType],
    arbitrary_types_allowed=True,
    json_encoders=JSON_ENCODERS,
    copy_on_model_validation=False,
    # Needed to avoid: https://github.com/samuelcolvin/pydantic/discussions/4099
):
    # executor_type must comply with https://peps.python.org/pep-3148/ standard
    loader: Optional[ObjLoader[ObjType]]  # loader of executor type

    # This is a utility field not included in reps. The typing lib has opened
    # issues on access of generic type within class.
    # This tracks for if-necessary future use.
    executor_type: type = Field(None, exclude=True)
    submit_callable: str = "submit"
    map_callable: str = "map"
    shutdown_callable: str = "shutdown"

    # executor will not be explicitely serialized, but loaded using loader with class
    # and kwargs
    executor: Optional[ObjType]

    @root_validator(pre=True)
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
            submit_callable = cls.__fields__["submit_callable"].default
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
            map_callable = cls.__fields__["map_callable"].default
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
            shutdown_callable = cls.__fields__["shutdown_callable"].default
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
        if executor_type not in cls.__config__.json_encoders:
            cls.__config__.json_encoders[
                executor_type
            ] = cls.__config__.json_encoders.pop(ObjType)

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
    arbitrary_types_allowed=True,
    json_encoders=JSON_ENCODERS,
):
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
        stored_kwargs = self.dict()

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
        "kwarg_order": Field(list(sig_kwargs.keys()), exclude=True),
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
