import datetime
import importlib
import inspect
import sys
import time
import traceback

import pandas as pd
import torch
import yaml

from .pydantic import get_descriptions_defaults
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


# functions for formatting documentation
def format_option_descriptions(options_object):
    options_dict = get_descriptions_defaults(options_object)
    return "\n\nGenerator Options\n" + yaml.dump(options_dict)


def read_xopt_csv(*files):
    """
    Read several Xopt-style CSV files into data

    Parameters
    ----------
    file1, file2, ...: path-like
        One or more Xopt csv files

    Returns
    -------
    pd.DataFrame
        DataFrame with xopt_index as the index column
    """
    dfs = []
    for file in files:
        df = pd.read_csv(file, index_col="xopt_index")
        dfs.append(df)
    return pd.concat(dfs)


def visualize_model(generator, data, axes=None):
    test_x = torch.linspace(*torch.tensor(generator.vocs.bounds.flatten()), 100)
    generator.add_data(data)
    model = generator.train_model()

    if axes is None:
        # Lazy import
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(2, 1, sharex="all")
        fig.set_size_inches(6, 6)
    else:
        ax = axes

    with torch.no_grad():
        post = model.posterior(test_x.reshape(-1, 1, 1).double())

        for i in range(len(generator.vocs.output_names)):
            mean = post.mean
            l, u = post.mvn.confidence_region()

            ax[0].plot(
                test_x,
                mean[..., i],
                f"C{i}",
                label=f"{generator.vocs.output_names[i]} model",
            )
            ax[0].fill_between(
                test_x, l[..., i].flatten(), u[..., i].flatten(), alpha=0.5
            )
            generator.data.plot(
                x=generator.vocs.variable_names[0],
                y=generator.vocs.output_names[i],
                ax=ax[0],
                style=f"oC{i}",
            )

        ax[0].legend()

        acq = generator.get_acquisition(model)(test_x.reshape(-1, 1, 1).double())

        ax[1].plot(test_x, acq, label="Acquisition Function")
        ax[1].legend()
