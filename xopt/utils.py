import datetime
import importlib
import inspect
import sys
import time
import traceback
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from pydantic import BaseModel

from xopt.generator import Generator
from .generators.bayesian.bayesian_generator import BayesianGenerator
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


def copy_generator(generator: Generator) -> Tuple[Generator, List[str]]:
    """
    Create a deep copy of a given generator.
    Moves any data saved on the gpu in the deepcopy of the generator to the cpu.

    Parameters
    ----------
    generator : Generator

    Returns
    -------
    generator_copy : Generator
    list_of_fields_on_gpu : list[str]
    """
    generator_copy = deepcopy(generator)
    generator_copy, list_of_fields_on_gpu = recursive_move_data_gpu_to_cpu(
        generator_copy
    )
    return generator_copy, list_of_fields_on_gpu


def recursive_move_data_gpu_to_cpu(
    pydantic_object: BaseModel,
) -> Tuple[BaseModel, List[str]]:
    """
    A recersive method to find all the data of a pydantic object
    which is stored on the gpu and then move that data to the cpu.

    Parameters
    ----------
    pydantic_object : BaseModel

    Returns
    -------
    pydantic_object : BaseModel
    list_of_fields_on_gpu : list[str]

    """
    pydantic_object_dict = pydantic_object.dict()
    list_of_fields_on_gpu = [pydantic_object.__class__.__name__]

    for field_name, field_value in pydantic_object_dict.items():
        if isinstance(field_value, BaseModel):
            result = recursive_move_data_gpu_to_cpu(field_value)
            pydantic_object_dict[field_name] = result[0]
            list_of_fields_on_gpu.append(result[1])
        if isinstance(field_value, torch.Tensor):
            if field_value.device.type == "cuda":
                pydantic_object_dict[field_name] = field_value.cpu()
                list_of_fields_on_gpu.append(field_name)
        elif isinstance(field_value, torch.nn.Module):
            if has_device_field(field_value, torch.device("cuda")):
                pydantic_object_dict[field_name] = field_value.cpu()
                list_of_fields_on_gpu.append(field_name)

    return pydantic_object, list_of_fields_on_gpu


def has_device_field(module: torch.nn.Module, device: torch.device) -> bool:
    """
    Checks if given module has a given device.

    Parameters
    ----------
    module : torch.nn.Module
    device : torch.device

    Returns
    -------
    True/False : bool
    """
    for parameter in module.parameters():
        if parameter.device == device:
            return True
    for buffer in module.buffers():
        if buffer.device == device:
            return True
    return False


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


def visualize_model(generator: BayesianGenerator, axes=None):
    test_x = torch.linspace(*generator.vocs.bounds.flatten(), 50).double()

    vocs = generator.vocs

    model = generator.train_model()

    # get acquisition function from generator
    acq = generator.get_acquisition(model)

    # calculate model posterior and acquisition function at each test point
    # NOTE: need to add a dimension to the input tensor for evaluating the
    # posterior and another for the acquisition function, see
    # https://botorch.org/docs/batching for details
    # NOTE: we use the `torch.no_grad()` environment to speed up computation by
    # skipping calculations for backpropagation
    with torch.no_grad():
        posterior = model.posterior(test_x.unsqueeze(1))
        acq_val = acq(test_x.reshape(-1, 1, 1))

    # get mean function and confidence regions
    mean = posterior.mean
    lower, upper = posterior.mvn.confidence_region()

    # plot model and acquisition function
    n_outputs = vocs.n_outputs
    if n_outputs == 1:
        lower = lower.unsqueeze(-1)
        upper = upper.unsqueeze(-1)

    if axes is None:
        # Lazy import
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(n_outputs + 1, 1, sharex="all")
        fig.set_size_inches(4, 2 * n_outputs)
    else:
        ax = axes
        fig = axes.gcf()

    # plot model posterior
    for i in range(n_outputs):
        ax[i].plot(test_x, mean[:, i], f"C{i}", label=vocs.output_names[i])
        ax[i].fill_between(test_x, lower[:, i], upper[:, i], alpha=0.25, fc=f"C{i}")

        # add data to model plot
        ax[i].plot(
            generator.data[vocs.variable_names],
            generator.data[vocs.output_names[i]],
            f"C{i}o",
            label="Training data",
        )
        ax[i].set_ylabel(vocs.output_names[i])

        if vocs.output_names[i] in vocs.constraint_names:
            ax[i].axhline(
                vocs.constraints[vocs.output_names[i]][-1], ls="--", c=f"C{i}"
            )

    # plot acquisition function
    ax[-1].plot(test_x, acq_val.flatten())

    ax[-1].set_ylabel(r"$\alpha(x)$")
    ax[-1].set_xlabel("x")

    return fig, ax


def get_local_region(center_point: dict, vocs:VOCS, fraction: float=0.1) -> dict:
    """
    calculates the bounds of a local region around a center point with side lengths
    equal to a fixed fraction of the input space for each variable

    """
    if not center_point.keys() == set(vocs.variable_names):
        raise KeyError("Center point keys must match vocs variable names")

    bounds = {}
    widths = {ele: vocs.variables[ele][1] - vocs.variables[ele][0] for ele in vocs.variable_names}

    for name in vocs.variable_names:
        bounds[name] = [
            center_point[name] - widths[name] * fraction,
            center_point[name] + widths[name] * fraction
        ]

    return bounds


def explode_all_columns(data: pd.DataFrame):
    """explode all data columns in dataframes that are lists or np.arrays"""
    list_types = []
    for name, val in data.iloc[0].items():
        if isinstance(val, list) or isinstance(val, np.ndarray):
            list_types += [name]

    if len(list_types):
        try:
            return data.explode(list_types)
        except ValueError:
            raise ValueError("evaluator outputs that are lists must match in size")
    else:
        return data
