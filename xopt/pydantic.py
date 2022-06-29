# globally modify pydantic base model to not allow extra keys and handle np arrays


import copy
import inspect
from concurrent.futures import Future
from importlib import import_module

import numpy as np
from pydantic import BaseModel, Field, root_validator, validate_arguments, validator, ValidationError, Extra
from pydantic import create_model, Field
from types import FunctionType, MethodType
from typing import Any, Callable, Dict, Generic, Iterable, Optional, TypeVar, Tuple

from pydantic.generics import GenericModel


ObjType = TypeVar("ObjType")

class XoptBaseModel(BaseModel):
    class Config:
        extra = "forbid"
        json_encoders = {
            np.ndarray: lambda x: x.tolist(),
            np.int64: lambda x: int(x),
            np.float64: lambda x: float(x),
        }

JSON_ENCODERS = {
    # function/method type distinguished for class members and not recognized as callables
    FunctionType: lambda x: f"{x.__module__}.{x.__qualname__}",
    MethodType: lambda x: f"{x.__module__}.{x.__qualname__}",
    Callable: lambda x: f"{x.__module__}.{type(x).__qualname__}",
    type: lambda x: f"{x.__module__}.{x.__name__}",
    # for encoding instances of the ObjType}
    ObjType: lambda x: f"{x.__module__}.{x.__class__.__qualname__}",
}




from pydantic import create_model, Field



class CallableModel(BaseModel):
    callable: Callable
    kwargs: BaseModel

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
        args = ()
        if "args" in values:
            args = values.pop("args")
            
        if "kwargs" in values:
            kwargs = values["kwargs"]

        # ignore kwarg-only and arg-only args for now
        sig_kwargs, _, _ = validate_and_compose_signature(callable, *args, **kwargs)
        
        # fix for pydantic handling...
        kwargs = {}
        for key, value in sig_kwargs.items():
            if isinstance(value, (tuple,)):
                kwargs[key] =(tuple, Field(None))
                
            elif value is None:
                kwargs[key] =(Any, Field(None))
                
            else:
                kwargs[key] = value  
        
        values["kwargs"] = create_model(f"Kwargs_{callable.__qualname__}", **kwargs)()
        
        return values
    

    def __call__(self, *args, **kwargs):
        if kwargs is None:
            kwargs = {}
            
        # create self.kwarg copy
        fn_kwargs = self.kwargs.dict()
        
        # update pos/kw kwargs with args
        if len(args):

            stored_kwargs = list(fn_kwargs.keys())

            for i, arg in enumerate(args[:len(fn_kwargs)]):
                fn_kwargs[stored_kwargs[i]] = arg
                
        # update stored kwargs
        fn_kwargs.update(kwargs)
        
        return self.callable(**fn_kwargs)




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
            # validate loader callable is same as obj type
            if values["loader"].get("callable") is not None:
                # unparameterized callable will handle parsing
                callable = CallableModel(
                    callable=values["loader"]["callable"]
                )
                
                if not callable.callable is obj_type:
                    raise ValueError(
                        "Provided loader of type %s. ObjLoader parameterized for %s",
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
    copy_on_model_validation = False, # Needed to avoid: https://github.com/samuelcolvin/pydantic/discussions/4099
):
    # executor_type must comply with https://peps.python.org/pep-3148/ standard
    loader: Optional[ObjLoader[ObjType]] # loader of executor type

    # This is a utility field not included in reps. The typing lib has opened issues on access of generic type within class.
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
        executor_type = cls.__fields__["executor"].type_ # introspect fields to get type

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
    except ModuleNotFoundError as err:
        logger.error("Unable to import module %s", module_name)
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


def validate_and_compose_signature(callable: Callable, *args, **kwargs):
    
    # try partial bind to validate
    signature = inspect.signature(callable)
    bound_args = signature.bind_partial(*args, **kwargs)
    
    sig_pos_or_kw = {}
    sig_kw_only = bound_args.arguments.get("kwargs")
    sig_args_only = bound_args.arguments.get("args")
    
    n_args = len(args)
    
    # Now go parameter by parameter and assemble kwargs
    for i, param in enumerate(signature.parameters.values()):

        if param.kind == param.POSITIONAL_OR_KEYWORD:
            sig_pos_or_kw[param.name] = param.default if not param.default == param.empty else None
            
            # assign via binding
            if param.name in bound_args.arguments:
                sig_pos_or_kw[param.name] = bound_args.arguments[param.name]
                
    
    return sig_pos_or_kw, sig_kw_only, sig_args_only        