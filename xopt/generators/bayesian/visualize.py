from typing import Any, Optional

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.models import ModelListGP
from pandas import DataFrame

from xopt.vocs import VOCS

from .objectives import feasibility


def visualize_generator_model(generator, **kwargs) -> tuple:
    """Displays GP model predictions for the selected output(s).

    The GP models are displayed with respect to the named variables. If None are given, the list of variables in
    generator.vocs is used. Feasible samples are indicated with a filled orange "o", infeasible samples with a
    hollow red "o". Feasibility is calculated with respect to all constraints unless the selected output is a
    constraint itself, in which case only that one is considered.

    Parameters
    ----------
    generator : BayesianGenerator
            Bayesian generator for which the GP model shall be visualized.
    **kwargs : visualization parameters
        See parameters of :func:`visualize_model`.

    Returns
    -------
    tuple
        The matplotlib figure and axes objects.
    """
    if generator.model is None:
        raise ValueError(
            "The generator.model doesn't exist, try calling generator.train_model()."
        )
    return visualize_model(
        model=generator.model,
        vocs=generator.vocs,
        data=generator.data,
        acquisition_function=generator.get_acquisition(generator.model),
        **kwargs,
    )


def visualize_model(
    model: ModelListGP,
    vocs: VOCS,
    data: DataFrame,
    acquisition_function: AcquisitionFunction = None,
    output_names: list[str] = None,
    variable_names: list[str] = None,
    idx: int = -1,
    reference_point: dict = None,
    show_samples: bool = True,
    show_prior_mean: bool = False,
    show_feasibility: bool = False,
    show_acquisition: bool = True,
    n_grid: int = 50,
    axes=None,
) -> tuple:
    """Displays GP model predictions for the selected output(s).

    The GP models are displayed with respect to the named variables. If None are given, the list of variables in
    vocs is used. Feasible samples are indicated with a filled orange "o", infeasible samples with a hollow
    red "o". Feasibility is calculated with respect to all constraints unless the selected output is a
    constraint itself, in which case only that one is considered.

    Parameters
    ----------
    model : ModelListGP
        GP model to visualize.
    vocs : VOCS
        VOCS corresponding to the GP model.
    data : DataFrame
        GP model data.
    acquisition_function : AcquisitionFunction, optional
        Acquisition function to visualize.
    output_names : List[str]
        Outputs for which the GP models are displayed. Defaults to all outputs in vocs.
    variable_names : List[str]
        The variables with respect to which the GP models are displayed (maximum of 2).
        Defaults to vocs.variable_names.
    idx : int
        Index of the last sample to use. This also selects the point of reference in
        higher dimensions unless an explicit reference_point is given.
    reference_point : dict
        Reference point determining the value of variables in vocs.variable_names, but not in variable_names
        (slice plots in higher dimensions). Defaults to last used sample.
    show_samples : bool, optional
        Whether samples are shown.
    show_prior_mean : bool, optional
        Whether the prior mean is shown.
    show_feasibility : bool, optional
        Whether the feasibility region is shown.
    show_acquisition : bool, optional
        Whether the acquisition function is computed and shown (only if acquisition function is not None).
    n_grid : int, optional
        Number of grid points per dimension used to display the model predictions.
    axes : Axes, optional
        Axes object used for plotting.

    Returns
    -------
    tuple
        The matplotlib figure and axes objects.
    """
    output_names, variable_names = _validate_names(output_names, variable_names, vocs)
    reference_point_names = [
        name for name in vocs.variable_names if name not in variable_names
    ]
    if show_acquisition and acquisition_function is None:
        show_acquisition = False
    kwargs = locals()
    dim_x, dim_y = len(variable_names), len(output_names)
    # plot configuration
    figure_config = _get_figure_config(min_ncols=dim_x, min_nrows=dim_y, **kwargs)
    if axes is None:
        from matplotlib import pyplot as plt  # lazy import

        fig, ax = plt.subplots(**figure_config)
    else:
        fig, ax = _get_figure_from_axes(axes), axes
    nrows, ncols = figure_config["nrows"], figure_config["ncols"]
    _verify_axes(ax, nrows, ncols)

    reference_point = _get_reference_point(reference_point, vocs, data, idx)

    figure_title = "Reference point: " + " ".join(
        [f"{name}: {reference_point[name]:.2}" for name in reference_point_names]
    )
    fig.suptitle(figure_title)

    # create plot
    if dim_x == 1:
        for i, output_name in enumerate(output_names):
            color_idx = 2 * i if i < 2 else i + 2
            plot_model_prediction(
                output_name=output_name,
                color=f"C{color_idx}",
                axis=ax[i],
                **kwargs,
            )
            ax[i].set_xlabel(None)
        if show_acquisition:
            plot_acquisition_function(
                axis=ax[len(output_names)], **(kwargs | {"show_samples": False})
            )
            ax[len(output_names)].set_xlabel(None)
        if show_feasibility:
            plot_feasibility(axis=ax[-1], **kwargs)
        ax[-1].set_xlabel(variable_names[0])
    else:
        # generate input mesh only once
        input_mesh = _generate_input_mesh(
            reference_point=reference_point,
            variable_names=variable_names,
            vocs=vocs,
            n_grid=n_grid,
        )
        for i, output_name in enumerate(output_names):
            posterior_mean, posterior_std, prior_mean = _get_model_predictions(
                input_mesh=input_mesh,
                output_name=output_name,
                model=model,
                vocs=vocs,
                include_prior_mean=show_prior_mean,
            )
            for j in range(ncols):
                ax_ij = ax[i, j] if nrows > 1 else ax[j]
                if j == 0:
                    prediction = posterior_mean
                    title = f"Posterior Mean [{output_name}]"
                    cbar_label = output_name
                elif j == 1:
                    prediction = posterior_std
                    title = f"Posterior SD [{output_name}]"
                    cbar_label = r"$\sigma\,$[{}]".format(output_name)
                else:
                    prediction = prior_mean
                    title = f"Prior Mean [{output_name}]"
                    cbar_label = output_name
                _plot2d_prediction(
                    prediction=prediction,
                    output_name=output_name,
                    input_mesh=input_mesh,
                    title=title,
                    cbar_label=cbar_label,
                    axis=ax_ij,
                    show_legend=i == j == 0,
                    **kwargs,
                )
        if show_acquisition:
            ax_acq = ax[len(output_names), 0]
            if hasattr(acquisition_function, "base_acquisition"):
                plot_acquisition_function(
                    axis=ax[len(output_names), 0],
                    only_base_acq=True,
                    show_legend=False,
                    **(kwargs | {"show_samples": False}),
                )
                ax_acq = ax[len(output_names), 1]
            else:
                ax[len(output_names), 1].axis("off")
            plot_acquisition_function(
                axis=ax_acq, show_legend=False, **(kwargs | {"show_samples": False})
            )
        if show_feasibility:
            if ncols == 3 and show_acquisition:
                ax_feasibility = ax[len(output_names), 2]
            elif ncols == 3 and not show_acquisition:
                ax_feasibility = ax[-1, 0]
                ax[-1, 1].axis("off")
                ax[-1, 2].axis("off")
            else:
                ax_feasibility = ax[-1, 0]
                ax[-1, 1].axis("off")
            plot_feasibility(
                axis=ax_feasibility,
                show_legend=False,
                **(kwargs | {"show_samples": False}),
            )
        else:
            if ncols == 3 and show_acquisition:
                ax[len(output_names), 2].axis("off")
        # set axis labels
        for i in range(nrows):
            for j in range(ncols):
                ax_ij = ax[i, j] if nrows > 1 else ax[j]
                ax_ij.set_xlabel(None)
                ax_ij.set_ylabel(None)
                if i == nrows - 1:
                    ax_ij.set_xlabel(variable_names[0])
                if j == 0:
                    ax_ij.set_ylabel(variable_names[1])
    fig.tight_layout()
    return fig, ax


def plot_model_prediction(
    model: ModelListGP,
    vocs: VOCS,
    data: DataFrame,
    output_name: str = None,
    variable_names: list[str] = None,
    prediction_type: str = None,
    idx: int = -1,
    reference_point: dict = None,
    show_samples: bool = True,
    show_prior_mean: bool = False,
    show_legend: bool = True,
    n_grid: int = 100,
    color: str = "C0",
    axis=None,
    **_,
):
    """Displays the GP model prediction for the selected output.

    Parameters
    ----------
    model : ModelListGP
        See eponymous parameter of :func:`visualize_model`.
    vocs : VOCS
        See eponymous parameter of :func:`visualize_model`.
    data : DataFrame
        See eponymous parameter of :func:`visualize_model`.
    output_name : str, optional
        Output for which the GP model prediction is displayed. Defaults to first output in vocs.
    variable_names : list[str], optional
        See eponymous parameter of :func:`visualize_model`.
    prediction_type : str, optional
        Determines the type of prediction to display ("posterior mean", "posterior std" or "prior mean").
        Defaults to "posterior mean".
    idx : int, optional
        See eponymous parameter of :func:`visualize_model`.
    reference_point : dict, optional
        See eponymous parameter of :func:`visualize_model`.
    show_samples : bool, optional
        See eponymous parameter of :func:`visualize_model`.
    show_prior_mean : bool, optional
        See eponymous parameter of :func:`visualize_model`.
    show_legend : bool, optional
        Whether to show the legend.
    n_grid : int, optional
        See eponymous parameter of :func:`visualize_model`.
    color : str, optional
        Color used for line plots.
    axis : Axes, optional
        The axis to use for plotting. If None is given, a new one is generated.
    _

    Returns
    -------
    Axes
        The axis.
    """
    if output_name is None:
        output_name = vocs.output_names[0]
    _, variable_names = _validate_names([output_name], variable_names, vocs)
    axis = _get_axis(axis, dim=len(variable_names))
    reference_point = _get_reference_point(reference_point, vocs, data, idx)
    kwargs = locals()
    input_mesh = _generate_input_mesh(**kwargs)
    requires_prior_mean = (
        prediction_type is not None and prediction_type.lower() == "prior mean"
    )
    posterior_mean, posterior_std, prior_mean = _get_model_predictions(
        input_mesh=input_mesh,
        output_name=output_name,
        model=model,
        vocs=vocs,
        include_prior_mean=show_prior_mean or requires_prior_mean,
    )
    if len(variable_names) == 1:
        x_axis = (
            input_mesh[:, vocs.variable_names.index(variable_names[0])]
            .squeeze()
            .numpy()
        )
        if output_name in vocs.constraint_names:
            axis.axhline(
                y=vocs.constraints[output_name][1],
                color=color,
                linestyle=":",
                label="Constraint Threshold",
            )
        if show_prior_mean:
            axis.plot(
                x_axis, prior_mean, color=color, linestyle="--", label="Prior Mean"
            )
        axis.plot(
            x_axis, posterior_mean, color=color, linestyle="-", label="Posterior Mean"
        )
        c = axis.fill_between(
            x=x_axis,
            y1=posterior_mean - 2 * posterior_std,
            y2=posterior_mean + 2 * posterior_std,
            color=color,
            alpha=0.25,
            label="",
        )
        if show_samples:
            plot_samples(**kwargs)
        # labels and legend
        axis.set_xlabel(variable_names[0])
        axis.set_ylabel(output_name)
        if show_legend:
            handles, labels = _combine_legend_entries_for_samples(
                *axis.get_legend_handles_labels()
            )
            for j in range(len(labels)):
                if labels[j] == "Posterior Mean":
                    labels[j] = r"Posterior Mean $\pm 2\,\sigma$"
                    handles[j] = (handles[j], c)
            from matplotlib.legend_handler import HandlerTuple  # lazy import

            axis.legend(
                labels=labels,
                handles=handles,
                handler_map={list: HandlerTuple(ndivide=None)},
            )
    else:
        prediction_type = (
            "posterior mean" if prediction_type is None else prediction_type
        )
        prediction_types = ["posterior mean", "posterior std", "prior mean"]
        if prediction_type.lower() not in prediction_types:
            raise ValueError(
                f"Unrecognized prediction type, must be one of {prediction_types}."
            )
        if prediction_type.lower() == "posterior mean":
            axis = _plot2d_prediction(
                prediction=posterior_mean,
                input_mesh=input_mesh,
                title=f"Posterior Mean [{output_name}]",
                cbar_label=output_name,
                **kwargs,
            )
        elif prediction_type.lower() == "posterior std":
            axis = _plot2d_prediction(
                prediction=posterior_std,
                input_mesh=input_mesh,
                title=f"Posterior SD [{output_name}]",
                cbar_label=r"$\sigma\,$[{}]".format(output_name),
                **kwargs,
            )
        else:
            axis = _plot2d_prediction(
                prediction=prior_mean,
                input_mesh=input_mesh,
                title=f"Prior Mean [{output_name}]",
                cbar_label=output_name,
                **kwargs,
            )
    return axis


def plot_acquisition_function(
    acquisition_function: AcquisitionFunction,
    vocs: VOCS,
    data: DataFrame,
    variable_names: list[str] = None,
    only_base_acq: bool = False,
    idx: int = -1,
    reference_point: dict = None,
    show_samples: bool = False,
    show_legend: bool = True,
    n_grid: int = 100,
    axis=None,
    **_,
):
    """Displays the given acquisition function.

    Parameters
    ----------
    acquisition_function : AcquisitionFunction
        The acquisition function to display.
    vocs : VOCS
        See eponymous parameter of :func:`visualize_model`.
    data : DataFrame
        See eponymous parameter of :func:`visualize_model`.
    variable_names : list[str], optional
        See eponymous parameter of :func:`visualize_model`.
    only_base_acq : bool, optional
        Whether to only plot the base acquisition function.
    idx : int, optional
        See eponymous parameter of :func:`visualize_model`.
    reference_point : dict, optional
        See eponymous parameter of :func:`visualize_model`.
    show_samples : bool, optional
        See eponymous parameter of :func:`visualize_model`.
    show_legend : bool, optional
        Whether to show the legend.
    n_grid : int, optional
        See eponymous parameter of :func:`visualize_model`.
    axis : Axes, optional
        The axis to use for plotting. If None is given, a new one is generated.
    _

    Returns
    -------
    Axes
        The axis.
    """
    _, variable_names = _validate_names(vocs.output_names, variable_names, vocs)
    axis = _get_axis(axis, dim=len(variable_names))
    reference_point = _get_reference_point(reference_point, vocs, data, idx)
    kwargs = locals()
    input_mesh = _generate_input_mesh(**kwargs)
    if len(variable_names) == 1:
        x_axis = (
            input_mesh[:, vocs.variable_names.index(variable_names[0])]
            .squeeze()
            .numpy()
        )
        base_acq = None
        if hasattr(acquisition_function, "base_acquisition"):
            base_acq = (
                acquisition_function.base_acquisition(input_mesh.unsqueeze(1))
                .detach()
                .squeeze()
                .numpy()
            )
        acq = acquisition_function(input_mesh.unsqueeze(1)).detach().squeeze().numpy()
        if base_acq is None:
            axis.plot(x_axis, acq, "C0-")
        else:
            axis.plot(x_axis, base_acq, "C0--", label="Base Acq. Function")
            if not only_base_acq:
                axis.plot(x_axis, acq, "C0-", label="Constrained Acq. Function")
            if show_samples:
                axis = plot_samples(**kwargs)
            if show_legend:
                axis.legend()
        axis.set_xlabel(variable_names[0])
        axis.set_ylabel(r"$\alpha\,$[{}]".format(vocs.output_names[0]))
    else:
        if only_base_acq:
            if not hasattr(acquisition_function, "base_acquisition"):
                raise ValueError(
                    "Given acquisition function doesn't have a base_acquisition attribute."
                )
            acq = (
                acquisition_function.base_acquisition(input_mesh.unsqueeze(1))
                .detach()
                .squeeze()
                .numpy()
            )
        else:
            acq = (
                acquisition_function(input_mesh.unsqueeze(1)).detach().squeeze().numpy()
            )
        if only_base_acq:
            title = "Base Acq. Function"
        elif hasattr(acquisition_function, "base_acquisition"):
            title = "Constrained Acq. Function"
        else:
            title = "Acq. Function"
        axis = _plot2d_prediction(
            prediction=acq,
            input_mesh=input_mesh,
            title=title,
            cbar_label=r"$\alpha\,$[{}]".format(vocs.output_names[0]),
            output_name=vocs.output_names[0],
            **kwargs,
        )
    return axis


def plot_feasibility(
    model: ModelListGP,
    vocs: VOCS,
    data: DataFrame,
    variable_names: list[str] = None,
    idx: int = -1,
    reference_point: dict = None,
    show_samples: bool = False,
    show_legend: bool = True,
    n_grid: int = 100,
    axis=None,
    **_,
):
    """Displays the feasibility region for the given model.

    Parameters
    ----------
    model : ModelListGP
        See eponymous parameter of :func:`visualize_model`.
    vocs : VOCS
        See eponymous parameter of :func:`visualize_model`.
    data : DataFrame
        See eponymous parameter of :func:`visualize_model`.
    variable_names : list[str], optional
        See eponymous parameter of :func:`visualize_model`.
    idx : int, optional
        See eponymous parameter of :func:`visualize_model`.
    reference_point : dict, optional
        See eponymous parameter of :func:`visualize_model`.
    show_samples : bool, optional
        See eponymous parameter of :func:`visualize_model`.
    show_legend : bool, optional
        Whether to show the legend.
    n_grid : int, optional
        See eponymous parameter of :func:`visualize_model`.
    axis : Axes, optional
        The axis to use for plotting. If None is given, a new one is generated.
    _

    Returns
    -------
    Axes
        The axis.
    """
    _, variable_names = _validate_names(vocs.output_names, variable_names, vocs)
    axis = _get_axis(axis, dim=len(variable_names))
    reference_point = _get_reference_point(reference_point, vocs, data, idx)
    kwargs = locals()
    input_mesh = _generate_input_mesh(**kwargs)
    feas = feasibility(input_mesh.unsqueeze(1), model, vocs).detach().squeeze().numpy()
    if len(variable_names) == 1:
        x_axis = (
            input_mesh[:, vocs.variable_names.index(variable_names[0])]
            .squeeze()
            .numpy()
        )
        axis.plot(x_axis, feas, "C0-")
        axis.set_xlabel(variable_names[0])
        axis.set_ylabel("Feasibility")
    else:
        axis = _plot2d_prediction(
            prediction=feas,
            output_name=vocs.output_names[0],
            input_mesh=input_mesh,
            title="Feasibility",
            cbar_label="Feasibility",
            **kwargs,
        )
    return axis


def plot_samples(
    vocs: VOCS,
    data: DataFrame,
    output_name: str = None,
    variable_names: list[str] = None,
    idx: int = -1,
    axis=None,
    **_,
):
    """Displays the data samples.

    Parameters
    ----------
    vocs : VOCS
        See eponymous parameter of :func:`visualize_model`.
    data : DataFrame
        See eponymous parameter of :func:`visualize_model`.
    output_name : str, optional
        The output used to determine the feasibility of a sample.
    variable_names : list[str], optional
        See eponymous parameter of :func:`visualize_model`.
    idx : int, optional
        See eponymous parameter of :func:`visualize_model`.
    axis : Axes, optional
        The axis to use for plotting. If None is given, a new one is generated.
    _

    Returns
    -------
    Axes
        The axis.
    """
    if output_name is None:
        output_name = vocs.output_names[0]
    _, variable_names = _validate_names([output_name], variable_names, vocs)
    axis = _get_axis(axis, dim=len(variable_names))
    kwargs = locals()
    x_feasible, y_feasible = _get_feasible_samples(**kwargs)
    if not x_feasible.size == 0:
        axis.scatter(
            x_feasible[:, 0] if len(variable_names) == 2 else x_feasible,
            x_feasible[:, 1] if len(variable_names) == 2 else y_feasible,
            marker="o",
            facecolors="C1",
            edgecolors="none",
            zorder=5,
            label="Feasible Samples",
        )
    x_infeasible, y_infeasible = _get_feasible_samples(**kwargs, reverse=True)
    if not x_infeasible.size == 0:
        axis.scatter(
            x_infeasible[:, 0] if len(variable_names) == 2 else x_infeasible,
            x_infeasible[:, 1] if len(variable_names) == 2 else y_infeasible,
            marker="o",
            facecolors="none",
            edgecolors="C3",
            zorder=5,
            label="Infeasible Samples",
        )
    axis.set_xlabel(variable_names[0])
    if len(variable_names) == 2:
        axis.set_ylabel(variable_names[1])
    else:
        axis.set_ylabel(output_name)
    return axis


def _plot2d_prediction(
    prediction: np.ndarray,
    vocs: VOCS,
    data: DataFrame,
    output_name: str,
    variable_names: list[str],
    input_mesh: torch.Tensor,
    title: str = None,
    cbar_label: str = None,
    show_samples: bool = True,
    show_legend: bool = True,
    n_grid: int = 100,
    axis=None,
    **_,
):
    """

    Parameters
    ----------
    prediction
    vocs : VOCS
        See eponymous parameter of :func:`visualize_model`.
    data : DataFrame
        See eponymous parameter of :func:`visualize_model`.
    output_name : str, optional
        Output for which the GP model prediction is displayed. Defaults to first output in vocs.
    variable_names : list[str], optional
        See eponymous parameter of :func:`visualize_model`.
    input_mesh : torch.Tensor
        Input mesh on which the GP model prediction is computed.
    title : str, optional
        Title of the generated plot.
    cbar_label : str, optional
        Label for the displayed colorbar.
    show_samples : bool, optional
        See eponymous parameter of :func:`visualize_model`.
    show_legend : bool, optional
        Whether to show the legend.
    n_grid : int, optional
        See eponymous parameter of :func:`visualize_model`.
    axis : Axes, optional
        The axis to use for plotting. If None is given, a new one is generated.
    _

    Returns
    -------
    Axes
        The axis.
    """
    axis = _get_axis(axis, dim=len(variable_names))
    kwargs = locals()
    axis.locator_params(axis="both", nbins=5)
    pcm = axis.pcolormesh(
        input_mesh[:, vocs.variable_names.index(variable_names[0])]
        .reshape(n_grid, n_grid)
        .numpy(),
        input_mesh[:, vocs.variable_names.index(variable_names[1])]
        .reshape(n_grid, n_grid)
        .numpy(),
        prediction.reshape(n_grid, n_grid),
        rasterized=True,
    )
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # lazy import

    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    from matplotlib import pyplot as plt  # lazy import

    cbar = plt.colorbar(pcm, cax=cax)
    axis.set_title(title)
    axis.set_xlabel(variable_names[0])
    axis.set_ylabel(variable_names[1])
    cbar.set_label(cbar_label)
    if show_samples:
        axis = plot_samples(**kwargs)
    if show_legend:
        handles, labels = _combine_legend_entries_for_samples(
            *axis.get_legend_handles_labels()
        )
        if handles or labels:
            from matplotlib.legend_handler import HandlerTuple  # lazy import

            axis.legend(
                labels=labels,
                handles=handles,
                handler_map={list: HandlerTuple(ndivide=None)},
            )
    return axis


def _generate_input_mesh(
    vocs: VOCS,
    variable_names: list[str],
    reference_point: dict[str, Any],
    n_grid: int,
    **_,
) -> torch.Tensor:
    """Generates an input mesh for visualization.

    Parameters
    ----------
    vocs : VOCS
        VOCS object for visualization.
    variable_names : List[str]
        Variable names with respect to which the GP model(s) shall be displayed.
    reference_point : dict
        Reference point determining the value of variables in vocs, but not in variable_names.
    n_grid : int
        Number of grid points per dimension used to generate the input mesh.
    _

    Returns
    -------
    torch.Tensor
        The input mesh for visualization.
    """
    x_lim = torch.tensor([vocs.variables[k] for k in variable_names])
    x_i = [torch.linspace(*x_lim[i], n_grid) for i in range(x_lim.shape[0])]
    x_mesh = torch.meshgrid(*x_i, indexing="ij")
    x_v = torch.hstack([ele.reshape(-1, 1) for ele in x_mesh]).double()
    x = torch.stack(
        [
            x_v[:, variable_names.index(k)]
            if k in variable_names
            else reference_point[k] * torch.ones(x_v.shape[0])
            for k in vocs.variable_names
        ],
        dim=-1,
    )
    return x


def _get_reference_point(
    reference_point: Optional[dict[str, Any]],
    vocs: VOCS,
    data: DataFrame,
    idx: int = -1,
) -> dict[str, Any]:
    """Returns a valid reference point.

    If the given reference point is None, the data sample corresponding to the given index is used.

    Parameters
    ----------
    reference_point : dict[str, Any] or None
        Reference point to validate. If not None, this is returned.
    vocs : VOCS
        VOCS object used for selecting variable names.
    data : DataFrame
        Data used to select a reference point.
    idx : int, optional
        Index of the sample to use as a reference point.

    Returns
    -------
    dict[str, Any]
        A valid reference point.
    """
    if reference_point is not None:
        return reference_point
    else:
        return data[vocs.variable_names].iloc[idx].to_dict()


def _get_model_predictions(
    model: ModelListGP,
    vocs: VOCS,
    output_name: str,
    input_mesh: torch.Tensor,
    include_prior_mean: bool = True,
    **_,
) -> tuple:
    """Returns the model predictions for the given output name and input mesh.

    Parameters
    ----------
    model : ModelListGP
        GP model used for predictions.
    vocs : VOCS
        VOCS object corresponding to the GP model.
    output_name : str
        Output name for which model predictions are computed.
    input_mesh : torch.Tensor
        Input mesh for which model predictions are computed.
    include_prior_mean : bool, optional
        Whether to include the prior mean in the predictions.
    _

    Returns
    -------
    tuple
        The model predictions.
    """
    gp = model.models[vocs.output_names.index(output_name)]
    with torch.no_grad():
        prior_mean = None
        if include_prior_mean:
            _x = gp.input_transform.transform(input_mesh)
            _x = gp.mean_module(_x)
            prior_mean = (
                gp.outcome_transform.untransform(_x)[0].detach().squeeze().numpy()
            )
        posterior = gp.posterior(input_mesh)
        posterior_mean = posterior.mean.detach().squeeze().numpy()
        posterior_std = torch.sqrt(posterior.mvn.variance).detach().squeeze().numpy()
    return posterior_mean, posterior_std, prior_mean


def _get_feasible_samples(
    vocs: VOCS,
    data: DataFrame,
    output_name: str,
    variable_names: list[str],
    idx: int = -1,
    reverse: bool = False,
    **_,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the feasible samples for the given output.

    Parameters
    ----------
    vocs : VOCS
        VOCS object used to compute feasibility.
    data : DataFrame
        Data tested for feasibility.
    output_name : str
        Output name for which feasibility and samples are returned.
    variable_names : List[str]
        Variable names used to select sample values.
    idx : int, optional
        Last data index to consider.
    reverse : bool, optional
        If True, the infeasible samples are returned instead.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (In-)feasible samples as a tuple of x and y values.
    """
    max_idx = idx + 1 if not idx == -1 else None
    if "feasible_" + output_name in vocs.feasibility_data(data).columns:
        feasible = vocs.feasibility_data(data).iloc[:max_idx]["feasible_" + output_name]
    else:
        feasible = vocs.feasibility_data(data).iloc[:max_idx]["feasible"]
    selector = feasible if not reverse else ~feasible
    x = data.iloc[:max_idx][variable_names][selector].to_numpy()
    y = data.iloc[:max_idx][output_name][selector].to_numpy()
    return x, y


def _validate_names(
    output_names: list[str],
    variable_names: list[str],
    vocs: VOCS,
) -> tuple[list[str], list[str]]:
    """Verifies that all names are in vocs and that the number of variable_names is valid.

    Parameters
    ----------
    output_names : List[str]
        Output names to verify.
    variable_names : List[str]
        Variable names to verify.
    vocs : VOCS
        VOCS object for verification.

    Returns
    -------
    tuple[list[str], list[str]]
        Validated output and variable names.
    """
    if output_names is None:
        output_names = vocs.output_names
    if variable_names is None:
        variable_names = vocs.variable_names
    if len(variable_names) not in [1, 2]:
        raise ValueError(
            f"Visualization is only supported with respect to 1 or 2 variables, not {len(variable_names)}. "
            f"Provide a compatible list of variable names to create slice plots at higher dimensions."
        )
    for names, s in zip(
        [output_names, variable_names], ["output_names", "variable_names"]
    ):
        invalid = [name not in getattr(vocs, s) for name in names]
        if any(invalid):
            invalid_names = [names[i] for i in range(len(names)) if invalid[i]]
            raise ValueError(f"Names {invalid_names} are not in vocs.{s}.")
    return output_names, variable_names


def _get_figure_config(
    min_ncols: int,
    min_nrows: int,
    show_acquisition: bool,
    show_prior_mean: bool,
    show_feasibility: bool,
    **_,
) -> dict[str, Any]:
    """Returns the matching plot configuration for model visualization.

    Parameters
    ----------
    min_ncols : int
        Minimum number of columns required.
    min_nrows : int
        Minimum number of rows required.
    show_acquisition : bool
        Whether the acquisition function will be displayed.
    show_prior_mean : bool
        Whether the prior mean will be displayed.
    show_feasibility : bool
        Whether the feasibility will be displayed.

    Returns
    -------
    dict[str, Any]
        Plot configuration for model visualization.
    """
    nrows, ncols = min_nrows, min_ncols
    if show_acquisition:
        nrows += 1
    if show_prior_mean and min_ncols == 2:
        ncols += 1
    if show_feasibility:
        if min_ncols == 2 and show_acquisition and show_prior_mean:
            pass
        else:
            nrows += 1
    if min_ncols == 1:
        sharex, sharey = True, False
        figsize = (6, 2 * nrows)
    else:
        sharex, sharey = True, True
        if nrows == 1:
            figsize = (4 * ncols, 3.7 * nrows)
        else:
            figsize = (4 * ncols, 3.3 * nrows)
    return {
        "nrows": nrows,
        "ncols": ncols,
        "sharex": sharex,
        "sharey": sharey,
        "figsize": figsize,
    }


def _get_figure_from_axes(axes):
    """Returns the figure corresponding to the given axes object.

    Parameters
    ----------
    axes : Axes
        The Axes object for which to return the figure.

    Returns
    -------
    Figure
        The figure corresponding to the given axes object.
    """
    from matplotlib.axes import Axes  # lazy import

    if isinstance(axes, Axes):
        return axes.get_figure()
    elif isinstance(axes, np.ndarray):
        ele = axes.flatten()[0]
        if isinstance(ele, Axes):
            return ele.get_figure()
        else:
            raise ValueError("Elements of multi-dimensional axes must be Axes objects.")
    else:
        raise ValueError(
            f"Expected Axes or np.ndarray object, but received {type(axes)}."
        )


def _get_axis(axis, dim: int = 1):
    """Returns a valid axis for plotting.

    If the given axis is None, a new Axes object is generated.

    Parameters
    ----------
    axis : Axes or None
        The axis to validate.
    dim : int, optional
        The plot dimension (determines default figure size).

    Returns
    -------
    Axes
        A valid axis.
    """
    from matplotlib.axes import Axes  # lazy import

    if isinstance(axis, Axes):
        return axis
    elif axis is None:
        figsize = _get_figure_config(dim, 1, False, False, False)["figsize"]
        import matplotlib.pyplot as plt  # lazy import

        fig, ax = plt.subplots(figsize=(figsize[0] / dim, figsize[1]))
        return ax
    else:
        raise ValueError(f"Expected Axes object or None, but received {type(axis)}.")


def _verify_axes(axes, nrows: int, ncols: int):
    """Verifies the given axes object has the correct type and shape.

    Parameters
    ----------
    axes : Axes
        The axes object to verify.
    nrows : int
        Expected number of rows.
    ncols : int
        Expected number of columns.
    """
    from matplotlib.axes import Axes  # lazy import

    if not (isinstance(axes, Axes) or isinstance(axes, np.ndarray)):
        raise ValueError(f"Expected Axes or np.ndarray, but received {type(axes)}.")
    axes_shape_is_valid = False
    if isinstance(axes, Axes) and nrows == ncols == 1:
        axes_shape_is_valid = True
    if isinstance(axes, np.ndarray):
        if len(axes.shape) == 1 and axes.shape[0] == nrows * ncols:
            axes_shape_is_valid = True
        if len(axes.shape) == 2 and axes.shape[0] == nrows and axes.shape[1] == ncols:
            axes_shape_is_valid = True
    if not axes_shape_is_valid:
        raise ValueError(
            f"Received Axes object does not match the expected shape ({nrows}, {ncols})."
        )


def _combine_legend_entries_for_samples(
    handles: list, labels: list
) -> tuple[list, list]:
    """Combines legend entries for feasible and infeasible samples.

    Parameters
    ----------
    handles : list
        Initial handles.
    labels : list
        Initial labels.

    Returns
    -------
    tuple[list, list]
        Updated handles and labels.
    """
    if all([ele in labels for ele in ["Feasible Samples", "Infeasible Samples"]]):
        labels[-2] = "In-/Feasible Samples"
        handles[-2] = [handles[-1], handles[-2]]
        del labels[-1], handles[-1]
    return handles, labels
