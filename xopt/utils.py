import datetime
import importlib
import inspect
import sys
import time
import traceback

import pandas as pd

from .vocs import VOCS


def add_constraint_information(data: pd.DataFrame, vocs: VOCS) -> pd.DataFrame:
    """
    determine if constraints have been satisfied based on data and vocs

    """
    temp_data = data.copy()

    # transform data s.t. negative values imply feasibility
    constraints = vocs.constraints

    for name, value in constraints.items():
        if value[0] == "GREATER_THAN":
            temp_data[name] = -(data[name] - value[1])
        else:
            temp_data[name] = data[name] - value[1]

        # add column to original dataframe
        data[f"{name}_feas"] = temp_data[name] < 0.0

    # add a column feas to show feasibility based on all of the constraints
    data["feas"] = data[[f"{ele}_feas" for ele in constraints]].all(axis=1)

    return data


def isotime(include_microseconds=False):
    """UTC to ISO 8601 with Local TimeZone information without microsecond"""
    t = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).astimezone()
    if not include_microseconds:
        t = t.replace(microsecond=0)

    return t.isoformat()


def get_function(name):
    """
    Returns a function from a fully qualified name or global name.
    """

    # Check if already a function
    if callable(name):
        return name

    if not isinstance(name, str):
        raise ValueError(f"{name} must be callable or a string.")

    if name in globals():
        if callable(globals()[name]):
            f = globals()[name]
        else:
            raise ValueError(f"global {name} is not callable")
    else:
        if "." in name:
            # try to import
            m_name, f_name = name.rsplit(".", 1)
            module = importlib.import_module(m_name)
            f = getattr(module, f_name)
        else:
            raise Exception(f"function {name} does not exist")

    return f


def get_function_defaults(f):
    """
    Returns a dict of the non-empty POSITIONAL_OR_KEYWORD arguments.

    See the `inspect` documentation for defaults.
    """
    defaults = {}
    for k, v in inspect.signature(f).parameters.items():
        if v.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            # print(k, v.default, v.kind)
            if v.default != inspect.Parameter.empty:
                defaults[k] = v.default
    return defaults


def get_n_required_fuction_arguments(f):
    """
    Counts the number of required function arguments using the `inspect` module.
    """
    n = 0
    for k, v in inspect.signature(f).parameters.items():
        if v.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if v.default == inspect.Parameter.empty:
                n += 1
    return n


def safe_call(func, *args, **kwargs):
    """
    Safely call the function, catching all exceptions.
    Returns a dict

    Parameters
    ----------
    func : Callable
        Function to call.
    args : tuple
        Arguments to pass to the function.
    kwargs : dict
        Keyword arguments to pass to the function.

    Returns
    -------
    outputs : dict
        result: result of the function call
        exception: exception raised by the function call
        traceback: traceback of the exception
        runtime: runtime of the function call in seconds
    """

    t = time.perf_counter()
    outputs = {}
    try:
        result = func(*args, **kwargs)
        outputs["exception"] = None
        outputs["traceback"] = ""
    except Exception:
        exc_tuple = sys.exc_info()
        error_str = traceback.format_exc()
        outputs = {}
        result = None
        outputs["exception"] = exc_tuple
        outputs["traceback"] = error_str
    finally:
        outputs["result"] = result
        outputs["runtime"] = time.perf_counter() - t
    return outputs
