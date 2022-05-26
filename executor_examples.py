import contextlib
import copy
import inspect
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from importlib import import_module
from typing import Any, Callable, Dict, Generic, Iterable, Optional, TypeVar

from pydantic import BaseModel, Field, root_validator, validate_arguments, validator
from pydantic.generics import GenericModel


logger = logging.getLogger("__name__")

ObjType = TypeVar("ObjType")

JSON_ENCODERS = {
    Callable: lambda x: f"{x.__module__}:{type(x).__name__}",
    type: lambda x: f"{x.__module__}:{x.__name__}",
    ObjType: lambda x: f"{x.__module__}:{x.__class__.__name__}",
}


@validate_arguments(config={"arbitrary_types_allowed": True})
def validate_and_compose_kwargs(signature: inspect.Signature, kwargs: Dict[str, Any]):

    required_kwargs = [
        kwarg.name
        for kwarg in signature.parameters.values()
        if (kwarg.POSITIONAL_OR_KEYWORD or kwarg.KEYWORD_ONLY)
        and kwarg.default is inspect.Parameter.empty
    ]

    if any([required_kwarg not in kwargs.keys() for required_kwarg in kwargs.keys()]):
        raise ValueError(
            "All required kwargs not provided: %s", ", ".join(required_kwargs)
        )

    # check (kwarg.VAR_KEYWORD and kwarg.default is inspect.Parameter.empty) is not empty **kwargs
    sig_kwargs = {
        kwarg.name: kwarg.default
        for kwarg in signature.parameters.values()
        if (kwarg.POSITIONAL_OR_KEYWORD or kwarg.KEYWORD_ONLY)
        and not kwarg.kind == inspect.Parameter.VAR_KEYWORD
    }

    # validate kwargs
    if any([kwarg not in sig_kwargs.keys() for kwarg in kwargs.keys()]):
        raise ValueError(
            "Kwargs must be members of function signature. Accepted kwargs are: %s, Provided: %s",
            ", ".join(sig_kwargs.keys()),
            ", ".join(kwargs.keys()),
        )

    sig_kwargs.update(kwargs)

    return sig_kwargs


class ParameterizedCallable(BaseModel):
    callable: Callable
    kwargs: dict

    class Config:
        arbitrary_types_allowed = True
        json_encoders = JSON_ENCODERS

    @root_validator(pre=True)
    def validate_all(cls, values):
        fn = values.pop("callable")

        if not isinstance(
            fn,
            (
                str,
                Callable,
            ),
        ):
            raise ValueError(
                "Callable must be object or a string. Provided %s", type(fn)
            )

        # parse string to callable
        if isinstance(fn, (str,)):

            # for function loading
            module_name, fn_name = fn.rsplit(":", 1)
            fn = getattr(import_module(module_name), fn_name)

        sig = inspect.signature(fn)

        # for reloading:
        if values.get("kwargs") is not None:
            values = values["kwargs"]

        kwargs = validate_and_compose_kwargs(sig, values)

        return {"callable": fn, "kwargs": kwargs}

    def call(self):
        return self.callable(**self.kwargs)


class UnparameterizedCallable(BaseModel):
    callable: Callable

    class Config:
        arbitrary_types_allowed = True
        json_encoders = JSON_ENCODERS

    @root_validator(pre=True)
    def validate_all(cls, values):
        fn = values.pop("callable")

        if not isinstance(
            fn,
            (
                str,
                Callable,
            ),
        ):
            raise ValueError(
                "Callable must be object or a string. Provided %s", type(fn)
            )

        # parse string to callables
        if isinstance(fn, (str,)):

            # for function loading
            module_name, fn_name = fn.rsplit(":", 1)
            fn = getattr(import_module(module_name), fn_name)

        return {"callable": fn}

    def call(self, *args, **kwargs):
        sig = inspect.signature(self.callable)
        kwargs = validate_and_compose_kwargs(sig, kwargs)
        return self.callable(**self.kwargs)


class ObjLoader(
    GenericModel,
    Generic[ObjType],
    arbitrary_types_allowed=True,
    json_encoders=JSON_ENCODERS,
):
    object: Optional[ObjType]
    loader: Optional[ParameterizedCallable]
    object_type: Optional[type]

    @root_validator(pre=True)
    def validate_all(cls, values):
        # inspect class init signature
        obj_type = cls.__fields__["object"].type_

        # adjust for re init from json
        if values.get("loader") is None:
            loader = ParameterizedCallable(callable=obj_type, **values)

        else:
            # validate loader callable is same as obj type
            if values["loader"].get("callable") is not None:
                # unparameterized callable will handle parsing
                callable = UnparameterizedCallable(
                    callable=values["loader"]["callable"]
                )

                if callable.callable != obj_type:
                    raise ValueError(
                        "Provided loader of type %s. ObjLoader parameterized for %s",
                        callable.callable.__name__,
                        obj_type,
                    )

                # opt for obj type
                values["loader"].pop("callable")

            # re-init drop callable from loader vals to use new instance
            loader = ParameterizedCallable(callable=obj_type, **values["loader"])

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
            return self.loader.call()


# COMMON BASE FOR EXECUTORS
class BaseExecutor(
    GenericModel,
    Generic[ObjType],
    arbitrary_types_allowed=True,
    json_encoders=JSON_ENCODERS,
):
    # executor_type must comply with https://peps.python.org/pep-3148/ standard
    loader: Optional[ObjLoader[ObjType]]
    executor_type: type = Field(None, exclude=True)
    submit_callable: str = "submit"
    map_callable: str = "map"
    shutdown_callable: str = "shutdown"

    # executor will not be explicitely serialized, but loaded using loader with class
    # and kwargs
    executor: Optional[ObjType]

    @root_validator(pre=True)
    def validate_all(cls, values):
        executor_type = cls.__fields__["executor"].type_

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

    def submit(self, fn, **kwargs) -> Future:
        # Create parameterized callable and sumbit
        submit_fn = getattr(self.executor, self.submit_callable)
        return submit_fn(fn, **kwargs)

    def map(self, fn, iter: Iterable) -> Iterable[Future]:
        map_fn = getattr(self.executor, self.map_callable)
        return map_fn(fn, iter)


# ContexExecutor with context handling on submission and no executor persistence
class ContextExecutor(
    BaseExecutor[ObjType],
    Generic[ObjType],
    arbitrary_types_allowed=True,
    json_encoders=JSON_ENCODERS,
):
    @contextlib.contextmanager
    def context(self):

        try:
            self.executor = self.loader.load()
            yield self.executor

        finally:
            self.shutdown()
            self.executor = None

    def submit(self, fn, **kwargs) -> Future:
        parameterized_fn = ParameterizedCallable(callable=fn, **kwargs)
        with self.context() as ctxt:
            submit_fn = getattr(ctxt, self.submit_callable)
            return submit_fn(parameterized_fn.call)

    def map(self, fn, iter: Iterable) -> Iterable[Future]:

        with self.context() as ctxt:
            map_fn = getattr(ctxt, self.map_callable)
            return map_fn(fn, iter)


class TestClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def test_function(x: int, y: int = 5):
    return x + y


if __name__ == "__main__":

    print("ParameterizedCallable using TestClass, x=1, y=3")
    parameterized_fn = ParameterizedCallable(callable=TestClass, x=1, y=3)
    result = parameterized_fn.call()
    print("Result of call:")
    print(result)
    parameterized_fn_dict = parameterized_fn.dict()
    print("Dict rep:")
    print(parameterized_fn_dict)
    parameterized_fn_from_dict = ParameterizedCallable(**parameterized_fn_dict)

    parameterized_fn_json = parameterized_fn.json()
    print("Json rep")
    print(parameterized_fn_json)
    print(parameterized_fn.callable)

    parameterized_fn_from_json = ParameterizedCallable.parse_raw(parameterized_fn_json)
    print("loaded new rep from json with:")
    print(f"kwargs: {parameterized_fn_from_json.kwargs}")
    print(f"callable: {parameterized_fn_from_json.callable}")

    print("Create object loader")
    obj_loader = ObjLoader[TestClass](x=1, y=3)
    loaded = obj_loader.load()
    print(loaded)
    print(f"Loaded object of type {type(loaded)} with x={loaded.x}, y={loaded.y}")
    print("json rep:")
    print(obj_loader.json())

    print("Testing ThreadPoolExecutor w/ object loader")
    tpe_loader = ObjLoader[ThreadPoolExecutor](max_workers=1)
    print("Loaded")

    print("ThreadPoolExecutor  w/ object loader to json")
    obj_loader_json = tpe_loader.json()
    print(obj_loader_json)
    ObjLoader[ThreadPoolExecutor].parse_raw(obj_loader_json)

    print("Creating threadpool ContextExecutor")
    context_exec = ContextExecutor[ThreadPoolExecutor](max_workers=1)
    print("Created.")

    print("Executor to json:")
    context_exec_json = context_exec.json()
    print(context_exec_json)

    print("Loading threadpool context executor from json")
    context_exec_from_json = ContextExecutor[ThreadPoolExecutor].parse_raw(
        context_exec_json
    )
    print("loaded new threadpool context executor from json")

    print("Calling threadpool context executor")
    future = context_exec.submit(fn=test_function, x=1, y=8)
    print(f"Returns: {future}")

    print("Run mapping function with threadpool context executor")
    futures = context_exec.map(test_function, ((1, 4), (3, 4)))
    print(f"Returns: {futures}")

    print("Loading threadpool normal executor")
    norm_exec = NormalExecutor[ThreadPoolExecutor](max_workers=1)
    print("Calling threadpool normal executor")
    future = norm_exec.submit(fn=test_function, x=1, y=8)
    print(f"Returns: {future}")

    print("Run mapping function with threadpool normal executor")
    futures = norm_exec.map(test_function, ((1, 4), (3, 4)))
    norm_exec.shutdown()
    print(f"Returns: {futures}")

    # DASK:
    from dask.distributed import Client

    client = Client(silence_logs=logging.ERROR)
    executor = client.get_executor()
    print("Creating dask normal executor w/ provided executor")
    dask_executor = NormalExecutor[type(executor)](executor=executor)
    print("Calling executor:")
    future = dask_executor.submit(fn=test_function, x=1, y=8)
    print(f"Returns: {future}")

    from dask.distributed.cfexecutor import ClientExecutor

    print("Creating dask normal executor w/ no provided executor")
    dask_executor = NormalExecutor[ClientExecutor](client=client)
    print("Calling executor:")
    future = dask_executor.submit(fn=test_function, x=1, y=8)
    print(f"Returns: {future}")

    norm_exec_json = norm_exec.json()
    print("Normal executor to json:")
    print(norm_exec_json)

    # wait for future completion till shutdown
    dask_executor.shutdown()

    print("Creating contextual dask executor")
    dask_executor = ContextExecutor[ClientExecutor](client=client)
    futures = dask_executor.map(test_function, ((1, 4), (3, 4)))
    print(f"Returns: {futures}")
