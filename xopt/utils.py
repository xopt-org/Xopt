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


def visualize_model(
        generator: BayesianGenerator,
        axes=None,
        output_name: str = None,
        variable_names: tuple[str, str] = None,
        train_model: bool = False,
        idx: int = -1,
        reference_point: dict = None,
        constrained_acqf: bool = True,
        n_grid: int = 50,
        figsize: tuple[float, float] = None,
        show_samples: bool = True,
        fading_samples: bool = True,
) -> tuple:
    """Displays GP model predictions for the selected output.

    The GP model is displayed with respect to the named variables. If None are given, the list of variables in
    generator.vocs is used. Feasible samples are indicated with orange "+"-marks, infeasible samples with red
    "o"-marks. Feasibility is calculated with respect to all constraints unless the selected output is a
    constraint itself, in which case only that one is considered.

    Args:
        generator: Bayesian generator object.
        axes: The matplotlib axes (if existing ones shall be used).
        output_name: Output for which the GP model is displayed.
        variable_names: The variables for which the model is displayed (maximum of 2).
          Defaults to generator.vocs.variable_names.
        train_model: Whether a new GP model shall be trained. Otherwise, the existing generator model is used.
        idx: Index of the last sample to use.
        reference_point: Reference point determining the value of variables in generator.vocs.variable_names,
          but not in variable_names. Defaults to last used sample.
        constrained_acqf: Determines whether the constrained or base acquisition function is shown.
        n_grid: Number of grid points per dimension used to display the model predictions.
        figsize: Size of the matplotlib figure. Defaults to (6, 4) for 1D and (10, 8) for 2D.
        show_samples: Determines whether samples are shown.
        fading_samples: Determines whether older samples are shown as more transparent.

    Returns:
        The matplotlib figure and axes objects.
    """

    # define output and variable names
    if output_name is None:
        output_name = generator.vocs.output_names[0]
    if variable_names is None:
        variable_names = generator.vocs.variable_names
    dim = len(variable_names)
    if dim not in [1, 2]:
        raise ValueError(f"Number of variables should be 1 or 2, not {dim}.")

    # generate input mesh
    if reference_point is None:
        reference_point = generator.data[generator.vocs.variable_names].iloc[idx].to_dict()
    x_lim = torch.tensor([generator.vocs.variables[k] for k in variable_names])
    x_i = [torch.linspace(*x_lim[i], n_grid) for i in range(x_lim.shape[0])]
    x_mesh = torch.meshgrid(*x_i, indexing="ij")
    x_v = torch.hstack([ele.reshape(-1, 1) for ele in x_mesh]).double()
    x = torch.stack(
        [x_v[:, variable_names.index(k)] if k in variable_names else reference_point[k] * torch.ones(x_v.shape[0])
         for k in generator.vocs.variable_names],
        dim=-1,
    )

    # compute model predictions
    if train_model:
        model = generator.train_model(update_internal=False)
        gp = model.models[generator.vocs.output_names.index(output_name)]
    else:
        model = generator.model
        gp = model.models[generator.vocs.output_names.index(output_name)]
    with torch.no_grad():
        _x = gp.input_transform.transform(x)
        _x = gp.mean_module(_x)
        prior_mean = gp.outcome_transform.untransform(_x)[0]
        posterior = gp.posterior(x)
        posterior_mean = posterior.mean
        posterior_sd = torch.sqrt(posterior.mvn.variance)
        if constrained_acqf:
            acqf_values = generator.get_acquisition(model)(x.unsqueeze(1))
        else:
            acqf_values = generator.get_acquisition(model).base_acqusition(x.unsqueeze(1))

    # determine feasible and infeasible samples
    max_idx = idx + 1
    if max_idx == 0:
        max_idx = None
    if "feasible_" + output_name in generator.vocs.feasibility_data(generator.data).columns:
        feasible = generator.vocs.feasibility_data(generator.data).iloc[:max_idx]["feasible_" + output_name]
    else:
        feasible = generator.vocs.feasibility_data(generator.data).iloc[:max_idx]["feasible"]
    feasible_samples = generator.data.iloc[:max_idx][variable_names][feasible]
    feasible_index = generator.data.iloc[:max_idx].index.values.astype(int)[feasible]
    infeasible_samples = generator.data.iloc[:max_idx][variable_names][~feasible]
    infeasible_index = generator.data.iloc[:max_idx].index.values.astype(int)[~feasible]
    idx_min = np.min(generator.data.iloc[:max_idx].index.values.astype(int))
    idx_max = np.max(generator.data.iloc[:max_idx].index.values.astype(int))
    alpha_min = 0.1  # if fading_samples

    # plot configuration
    if dim == 1:
        sharex = True
        nrows, ncols = 2, 1
        if figsize is None:
            figsize = (6, 4)
    else:
        sharex = False
        nrows, ncols = 2, 2
        if figsize is None:
            figsize = (10, 8)

    # create figure and axes if required
    if axes is None:
        # lazy import
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex)
    else:
        if isinstance(axes, np.ndarray):
            fig = axes.flatten()[0].get_figure()
        else:
            fig = axes.get_figure()

    # plot data
    z = [posterior_mean, prior_mean, posterior_sd, acqf_values]
    labels = ["Posterior Mean", "Prior Mean", "Posterior SD"]
    if dim == 1:
        labels[2] = "Posterior CL ($\pm 2\,\sigma$)"
    if constrained_acqf:
        labels.append("Constrained Acquisition Function")
    else:
        labels.append("Base Acquisition Function")

    for i in range(nrows * ncols):
        ax = axes.flatten()[i]

        # base plot
        if dim == 1:
            x_axis = x[:, generator.vocs.variable_names.index(variable_names[0])].squeeze().numpy()
            if i == 0:
                ax.plot(x_axis, z[1].detach().squeeze().numpy(), "C2--", label=labels[1])
                ax.plot(x_axis, z[0].detach().squeeze().numpy(), "C0", label=labels[0])
                ax.fill_between(x_axis, z[0].detach().squeeze().numpy() - 2 * z[2].detach().squeeze().numpy(),
                                z[0].detach().squeeze().numpy() + 2 * z[2].detach().squeeze().numpy(),
                                color="C0", alpha=0.25, label=labels[2])
                ax.set_ylabel(output_name)
            else:
                ax.plot(x_axis, z[-1].detach().squeeze().numpy(), label=labels[-1])
                ax.set_xlabel(variable_names[0])
                ax.set_ylabel(r"$\alpha\,$[{}]".format(generator.vocs.output_names[0]))
            ax.legend()
        else:
            pcm = ax.pcolormesh(x_mesh[0].numpy(), x_mesh[1].numpy(),
                                z[i].detach().squeeze().reshape(n_grid, n_grid).numpy())
            ax.locator_params(axis="both", nbins=5)
            ax.set_title(labels[i])
            ax.set_xlabel(variable_names[0])
            if i % nrows == 0:
                ax.set_ylabel(variable_names[1])
            cbar = fig.colorbar(pcm, ax=ax)
            if i == 2:
                cbar_label = r"$\sigma\,$[{}]".format(output_name)
            elif i == 3:
                cbar_label = r"$\alpha\,$[{}]".format(generator.vocs.output_names[0])
            else:
                cbar_label = output_name
            cbar.set_label(cbar_label)

        # plot samples
        if show_samples:
            x_0_feasible, x_1_feasible = None, None
            x_0_infeasible, x_1_infeasible = None, None
            if dim == 1 and i == 0:
                if not feasible_samples.empty:
                    x_0_feasible = feasible_samples.to_numpy()
                    x_1_feasible = generator.data.iloc[:max_idx][output_name][feasible].to_numpy()
                if not infeasible_samples.empty:
                    x_0_infeasible = infeasible_samples.to_numpy()
                    x_1_infeasible = generator.data.iloc[:max_idx][output_name][~feasible].to_numpy()
            elif dim == 2:
                if not feasible_samples.empty:
                    x_0_feasible, x_1_feasible = feasible_samples.to_numpy().T
                if not infeasible_samples.empty:
                    x_0_infeasible, x_1_infeasible = infeasible_samples.to_numpy().T
            if x_0_feasible is not None and x_1_feasible is not None:
                if fading_samples and idx_min < idx_max:
                    for j in range(len(feasible_index)):
                        alpha = alpha_min + (1 - alpha_min) * ((feasible_index[j] - idx_min) / (idx_max - idx_min))
                        ax.scatter(x_0_feasible[j], x_1_feasible[j], marker="+", c="C1", alpha=alpha)
                else:
                    ax.scatter(x_0_feasible, x_1_feasible, marker="+", c="C1")
            if x_0_infeasible is not None and x_1_infeasible is not None:
                if fading_samples and idx_min < idx_max:
                    for j in range(len(infeasible_index)):
                        alpha = alpha_min + (1 - alpha_min) * (
                                    (infeasible_index[j] - idx_min) / (idx_max - idx_min))
                        ax.scatter(x_0_infeasible[j], x_1_infeasible[j], marker="o", c="C3", alpha=alpha)
                else:
                    ax.scatter(x_0_infeasible, x_1_infeasible, marker="o", c="C3")

    fig.tight_layout()
    return fig, axes


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
            center_point[name] - widths[name] * fraction,
            center_point[name] + widths[name] * fraction,
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
            return data.explode(list_types, ignore_index=True)
        except ValueError:
            raise ValueError("evaluator outputs that are lists must match in size")
    else:
        return data
