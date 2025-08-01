from copy import deepcopy
from pydantic import BaseModel
from typing import List, Tuple
import datetime
import importlib
import inspect
import logging
import numpy as np
import os
import pandas as pd
import sys
import time
import torch
import traceback
import yaml

from .generator import Generator
from .pydantic import get_descriptions_defaults
from .vocs import VOCS


# Grab the logger
logger = logging.getLogger(__name__)


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
    pydantic_object_dict = pydantic_object.model_dump()
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


def get_local_region(center_point: dict, vocs: VOCS, fraction: float = 0.1) -> dict:
    """
    calculates the bounds of a local region around a center point with side lengths
    equal to a fixed fraction of the input space for each variable

    """
    if not center_point.keys() == set(vocs.variable_names):
        raise KeyError("Center point keys must match vocs variable names")

    bounds = {}
    widths = {
        ele: vocs.variables[ele][1] - vocs.variables[ele][0]
        for ele in vocs.variable_names
    }

    for name in vocs.variable_names:
        bounds[name] = [
            np.max(
                (center_point[name] - widths[name] * fraction, vocs.variables[name][0])
            ),
            np.min(
                (center_point[name] + widths[name] * fraction, vocs.variables[name][1])
            ),
        ]

    return bounds


########################################################################################################################
# Data processing
########################################################################################################################


def read_csv(filepath: str, last_n_lines: int | None = None, **kwargs) -> pd.DataFrame:
    """
    Wrapper for pandas.read_csv with addition of only reading last n lines.
    """
    if last_n_lines is not None:
        # Count total lines (subtract 1 for header)
        with open(filepath, "r") as f:
            total_lines = sum(1 for _ in f) - 1

        # Calculate how many lines to skip
        skiprows = max(0, total_lines - last_n_lines)
        skiprows = range(1, skiprows + 1)
    else:
        skiprows = None

    # Read with skiprows
    return pd.read_csv(filepath, skiprows=skiprows, **kwargs)


def nsga2_to_cnsga_file_format(
    input_dir: str, output_dir: str, last_n_lines: int | None = None
):
    """
    Convert the output of the NSGA2 generator to the same format used by the CNSGA generator. This
    function is useful for interfacing with existing analysis tools.

    The converted output is guaranteed to be reproducible for the same input data. To this end, the
    converted filenames follow the format from the CNSGA generator `cnsga_population_<timestamp>.csv`
    and `cnsga_offspring_<timtestamp>.csv` where the timestamp is the generation index in seconds since
    epoch.

    Parameters:
    -----------
    input_dir : str
        The output directory of the NSGA2 generator.
    output_dir : str
        Where the converted output will be saved. Directory will be created if necessary.
    last_n_lines : int
        Read only the last n lines of each CSV file (useful for pulling final generations in large files)
    """
    # Load the population and data files
    pop = read_csv(
        os.path.join(input_dir, "populations.csv"), last_n_lines=last_n_lines
    )
    dat = read_csv(os.path.join(input_dir, "data.csv"), last_n_lines=last_n_lines)

    # Setup the output dir
    os.makedirs(output_dir, exist_ok=True)

    # Write each population (separated by key `xopt_generation`)
    generations = 0
    for generation in pop["xopt_generation"].unique():
        # Filter population data for this generation
        gen_pop = pop[pop["xopt_generation"] == generation]

        # Build filename
        timestamp = (
            datetime.datetime.fromtimestamp(int(generation), tz=datetime.timezone.utc)
            .isoformat()
            .replace(":", "_")
        )
        filename = f"cnsga_population_{timestamp}.csv"
        filename = os.path.join(output_dir, filename)

        # Write generation data to file
        gen_pop.to_csv(filename, index_label="xopt_index")
        logger.debug(
            f'Saved population file for generation {generation} to "{filename}"'
        )
        generations += 1

    # Write each set of the offspring (separated by key `xopt_parent_generation`)
    offsprings = 0
    for generation in dat["xopt_parent_generation"].unique():
        # Filter population data for this generation
        gen_pop = dat[dat["xopt_parent_generation"] == generation]

        # Build the filename
        # Note: to match CNSGA generator behavior where the candidates just received by the generator
        # have the same timestamp as the completed generation, the `xopt_parent_generation` needs
        # additional factor of one.
        timestamp = (
            datetime.datetime.fromtimestamp(
                int(generation) + 1, tz=datetime.timezone.utc
            )
            .isoformat()
            .replace(":", "_")
        )
        filename = f"cnsga_offspring_{timestamp}.csv"
        filename = os.path.join(output_dir, filename)

        # Write generation data to file
        gen_pop.to_csv(filename, index_label="xopt_index")
        logging.debug(
            f'Saved offspring file for generation {generation} to "{filename}"'
        )
        offsprings += 1

    # Some logging
    logging.info(
        f'Converted NSGA2Generator output "{input_dir}" to CNSGA2Generator format at "{output_dir}" ({generations} population files, {offsprings} offspring files, last_n_lines={last_n_lines})'
    )


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


def explode_all_columns(data: pd.DataFrame):
    """explode all data columns in dataframes that are lists or np.arrays"""
    # TODO: rework the whole list return type handling - this is really slow
    list_types = []
    lengths = []
    for name, val in data.iloc[0].items():
        if isinstance(val, (list, np.ndarray)):
            list_types.append(name)
            try:
                lengths.append(len(val))
            except TypeError:
                # handle case when a zero length ndarray is passed
                lengths.append(1)
    if len(list_types) > 0:
        if len(set(lengths)) > 1:
            raise ValueError("evaluator outputs that are lists must match in size")

        if data.shape[0] == 1:
            # Fast path for most common experimental case of 1 candidate per step
            df = _explode_pandas_modified(data, list_types, lengths[0])
            return df
        else:
            if len(list_types):
                try:
                    # dtype of return is object, but we have floats...
                    # https://github.com/pandas-dev/pandas/issues/34923
                    # also, this method is implemented in Python and uses slow calls
                    return data.explode(list_types, ignore_index=True)
                except ValueError:
                    raise ValueError(
                        "evaluator outputs that are lists must match in size"
                    )
    else:
        return data


def _explode_pandas_modified(df: pd.DataFrame, columns: list[str], length: int):
    if len(df) != 1:
        raise NotImplementedError("This method only works for single row dataframes")
    # this is slower somehow
    # df.to_dict(orient='records')
    data = {c: df[c].iloc[0] if c in columns else [df[c].iloc[0]] for c in df.columns}
    result = pd.DataFrame(data, index=np.arange(length))

    return result
