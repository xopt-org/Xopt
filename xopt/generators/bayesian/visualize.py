import torch

from .objectives import feasibility


def visualize_generator_model(
    generator,
    output_names: list[str] = None,
    variable_names: list[str] = None,
    idx: int = -1,
    reference_point: dict = None,
    show_samples: bool = True,
    show_prior_mean: bool = False,
    show_feasibility: bool = False,
    show_acquisition: bool = True,
    n_grid: int = 50,
) -> tuple:
    """Displays GP model predictions for the selected output(s).

    The GP models are displayed with respect to the named variables. If None are given, the list of variables in
    generator.vocs is used. Feasible samples are indicated with a filled orange "o", infeasible samples with a
    hollow red "o". Feasibility is calculated with respect to all constraints unless the selected output is a
    constraint itself, in which case only that one is considered.

    Args:
        generator: Bayesian generator object.
        output_names: Outputs for which the GP models are displayed. Defaults to all outputs in generator.vocs.
        variable_names: The variables with respect to which the GP models are displayed (maximum of 2).
          Defaults to generator.vocs.variable_names.
        idx: Index of the last sample to use. This also selects the point of reference in higher dimensions unless
          an explicit reference_point is given.
        reference_point: Reference point determining the value of variables in generator.vocs.variable_names,
          but not in variable_names (slice plots in higher dimensions). Defaults to last used sample.
        show_samples: Whether samples are shown.
        show_prior_mean: Whether the prior mean is shown.
        show_feasibility: Whether the feasibility region is shown.
        show_acquisition: Whether the acquisition function is computed and shown.
        n_grid: Number of grid points per dimension used to display the model predictions.

    Returns:
        The matplotlib figure and axes objects.
    """

    # define output and variable names
    vocs, data = generator.vocs, generator.data
    if output_names is None:
        output_names = vocs.output_names
    if variable_names is None:
        variable_names = vocs.variable_names
    dim_x, dim_y = len(variable_names), len(output_names)
    if dim_x not in [1, 2]:
        raise ValueError(f"Number of variables should be 1 or 2, not {dim_x}.")

    # check names exist in vocs
    for names, s in zip(
        [output_names, variable_names], ["output_names", "variable_names"]
    ):
        invalid = [name not in getattr(vocs, s) for name in names]
        if any(invalid):
            invalid_names = [names[i] for i in range(len(names)) if invalid[i]]
            raise ValueError(f"Names {invalid_names} are not in generator.vocs.{s}.")

    # generate input mesh
    if reference_point is None:
        reference_point = data[vocs.variable_names].iloc[idx].to_dict()
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

    # compute model predictions
    if generator.model is None:
        raise ValueError(
            "The generator.model doesn't exist, try calling generator.train_model()."
        )
    model = generator.model
    predictions = {}
    for output_name in output_names:
        gp = model.models[vocs.output_names.index(output_name)]
        with torch.no_grad():
            prior_mean = None
            if show_prior_mean:
                _x = gp.input_transform.transform(x)
                _x = gp.mean_module(_x)
                prior_mean = (
                    gp.outcome_transform.untransform(_x)[0].detach().squeeze().numpy()
                )
            posterior = gp.posterior(x)
            posterior_mean = posterior.mean.detach().squeeze().numpy()
            posterior_sd = torch.sqrt(posterior.mvn.variance).detach().squeeze().numpy()
        predictions[output_name] = [posterior_mean, posterior_sd, prior_mean]
    # acquisition function
    if show_acquisition:
        base_acq = None
        acq = generator.get_acquisition(model)
        if hasattr(acq, "base_acquisition"):
            base_acq = acq.base_acquisition(x.unsqueeze(1)).detach().squeeze().numpy()
        predictions["acq"] = [base_acq, acq(x.unsqueeze(1)).detach().squeeze().numpy()]
    if show_feasibility:
        predictions["feasibility"] = (
            feasibility(x.unsqueeze(1), model, vocs).detach().squeeze().numpy()
        )

    # determine feasible and infeasible samples
    max_idx = idx + 1
    if max_idx == 0:
        max_idx = None
    samples = {}
    for output_name in output_names:
        if "feasible_" + output_name in vocs.feasibility_data(data).columns:
            feasible = vocs.feasibility_data(data).iloc[:max_idx][
                "feasible_" + output_name
            ]
        else:
            feasible = vocs.feasibility_data(data).iloc[:max_idx]["feasible"]
        feasible_samples = data.iloc[:max_idx][variable_names][feasible]
        infeasible_samples = data.iloc[:max_idx][variable_names][~feasible]
        samples[output_name] = [feasible, feasible_samples, infeasible_samples]

    # plot configuration
    nrows, ncols = dim_y, dim_x
    if show_acquisition:
        nrows += 1
    if show_prior_mean and dim_x == 2:
        ncols += 1
    if show_feasibility:
        if dim_x == 2 and show_acquisition and show_prior_mean:
            pass
        else:
            nrows += 1
    if dim_x == 1:
        sharex, sharey = True, False
        figsize = (6, 2 * nrows)
    else:
        sharex, sharey = True, True
        if nrows == 1:
            figsize = (4 * ncols, 3.7 * nrows)
        else:
            figsize = (4 * ncols, 3.2 * nrows)
    # lazy import
    from matplotlib import pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, figsize=figsize
    )

    # create plot
    if dim_x == 1:
        x_axis = x[:, vocs.variable_names.index(variable_names[0])].squeeze().numpy()
        for i, output_name in enumerate(output_names):
            # model predictions
            if i == 0:
                color_idx = 0
            elif i == 1:
                color_idx = 2
            else:
                color_idx = i + 2
            if output_name in vocs.constraint_names:
                ax[i].axhline(
                    y=vocs.constraints[output_name][1],
                    color=f"C{color_idx}",
                    linestyle=":",
                    label="Constraint Threshold",
                )
            if show_prior_mean:
                ax[i].plot(
                    x_axis,
                    predictions[output_name][2],
                    f"C{color_idx}--",
                    label="Prior Mean",
                )
            ax[i].plot(
                x_axis,
                predictions[output_name][0],
                f"C{color_idx}-",
                label="Posterior Mean",
            )
            c = ax[i].fill_between(
                x_axis,
                predictions[output_name][0] - 2 * predictions[output_name][1],
                predictions[output_name][0] + 2 * predictions[output_name][1],
                color=f"C{color_idx}",
                alpha=0.25,
                label="",
            )
            # data samples
            if show_samples:
                if not samples[output_name][1].empty:
                    x_feasible = samples[output_name][1].to_numpy()
                    y_feasible = data.iloc[:max_idx][output_name][
                        samples[output_name][0]
                    ].to_numpy()
                    ax[i].scatter(
                        x_feasible,
                        y_feasible,
                        marker="o",
                        facecolors="C1",
                        edgecolors="none",
                        zorder=5,
                        label="Feasible Samples",
                    )
                if not samples[output_name][2].empty:
                    x_infeasible = samples[output_name][2].to_numpy()
                    y_infeasible = data.iloc[:max_idx][output_name][
                        ~samples[output_name][0]
                    ].to_numpy()
                    ax[i].scatter(
                        x_infeasible,
                        y_infeasible,
                        marker="o",
                        facecolors="none",
                        edgecolors="C3",
                        zorder=5,
                        label="Infeasible Samples",
                    )
            ax[i].set_ylabel(output_name)
            handles, labels = ax[i].get_legend_handles_labels()
            for j in range(len(labels)):
                if labels[j] == "Posterior Mean":
                    labels[j] = r"Posterior Mean $\pm 2\,\sigma$"
                    handles[j] = (handles[j], c)
            if all(
                [ele in labels for ele in ["Feasible Samples", "Infeasible Samples"]]
            ):
                labels[-2] = "In-/Feasible Samples"
                handles[-2] = [handles[-1], handles[-2]]
                del labels[-1], handles[-1]
            ax[i].legend(
                labels=labels,
                handles=handles,
                handler_map={list: HandlerTuple(ndivide=None)},
            )
        # acquisition function
        if not show_acquisition:
            pass
        else:
            if predictions["acq"][0] is None:
                ax[len(output_names)].plot(x_axis, predictions["acq"][1], "C0-")
            else:
                ax[len(output_names)].plot(
                    x_axis, predictions["acq"][0], "C0--", label="Base Acq. Function"
                )
                ax[len(output_names)].plot(
                    x_axis, predictions["acq"][1], "C0-", label="Constrained Acq. Function"
                )
                ax[len(output_names)].legend()
            ax[len(output_names)].set_ylabel(r"$\alpha\,$[{}]".format(vocs.output_names[0]))
        # feasibility
        if show_feasibility:
            ax[-1].plot(x_axis, predictions["feasibility"], "C0-")
            ax[-1].set_ylabel("Feasibility")
        ax[-1].set_xlabel(variable_names[0])

    else:
        for i in range(nrows):
            for j in range(ncols):
                ax_ij = ax[i, j] if nrows > 1 else ax[j]
                if i == nrows - 1:
                    ax_ij.set_xlabel(variable_names[0])
                if j == 0:
                    ax_ij.set_ylabel(variable_names[1])
                ax_ij.locator_params(axis="both", nbins=5)
        for i, output_name in enumerate(output_names):
            for j in range(ncols):
                ax_ij = ax[i, j] if nrows > 1 else ax[j]
                # model predictions
                pcm = ax_ij.pcolormesh(
                    x_mesh[0].numpy(),
                    x_mesh[1].numpy(),
                    predictions[output_name][j].reshape(n_grid, n_grid),
                )
                divider = make_axes_locatable(ax_ij)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = fig.colorbar(pcm, cax=cax)
                if j == 0:
                    ax_ij.set_title(f"Posterior Mean [{output_name}]")
                    cbar.set_label(output_name)
                elif j == 1:
                    ax_ij.set_title(f"Posterior SD [{output_name}]")
                    cbar.set_label(r"$\sigma\,$[{}]".format(output_name))
                else:
                    ax_ij.set_title(f"Prior Mean [{output_name}]")
                    cbar.set_label(output_name)
                # data samples
                if show_samples:
                    if not samples[output_name][1].empty:
                        x1_feasible, x2_feasible = samples[output_name][1].to_numpy().T
                        ax_ij.scatter(
                            x1_feasible,
                            x2_feasible,
                            marker="o",
                            facecolors="C1",
                            edgecolors="none",
                            zorder=5,
                            label="Feasible Samples",
                        )
                    if not samples[output_name][2].empty:
                        x1_infeasible, x2_infeasible = (
                            samples[output_name][2].to_numpy().T
                        )
                        ax_ij.scatter(
                            x1_infeasible,
                            x2_infeasible,
                            marker="o",
                            facecolors="none",
                            edgecolors="C3",
                            zorder=5,
                            label="Infeasible Samples",
                        )
                if i == j == 0:
                    handles, labels = ax_ij.get_legend_handles_labels()
                    if all(
                        [
                            ele in labels
                            for ele in ["Feasible Samples", "Infeasible Samples"]
                        ]
                    ):
                        labels[-2] = "In-/Feasible Samples"
                        handles[-2] = [handles[-1], handles[-2]]
                        del labels[-1], handles[-1]
                    ax_ij.legend(
                        labels=labels,
                        handles=handles,
                        handler_map={list: HandlerTuple(ndivide=None)},
                    )
        # acquisition function
        if not show_acquisition:
            pass
        else:
            if predictions["acq"][0] is None:
                pcm = ax[len(output_names), 0].pcolormesh(
                    x_mesh[0].numpy(),
                    x_mesh[1].numpy(),
                    predictions["acq"][1].reshape(n_grid, n_grid),
                )
                divider = make_axes_locatable(ax[len(output_names), 0])
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = fig.colorbar(pcm, cax=cax)
                cbar.set_label(r"$\alpha\,$[{}]".format(vocs.output_names[0]))
                ax[len(output_names), 0].set_title("Acq. Function")
                ax[len(output_names), 1].axis("off")
            else:
                for j in range(2):
                    pcm = ax[len(output_names), j].pcolormesh(
                        x_mesh[0].numpy(),
                        x_mesh[1].numpy(),
                        predictions["acq"][j].reshape(n_grid, n_grid),
                    )
                    divider = make_axes_locatable(ax[len(output_names), j])
                    cax = divider.append_axes("right", size="5%", pad=0.1)
                    cbar = fig.colorbar(pcm, cax=cax)
                    cbar.set_label(r"$\alpha\,$[{}]".format(vocs.output_names[0]))
                ax[len(output_names), 0].set_title("Base Acq. Function")
                ax[len(output_names), 1].set_title("Constrained Acq. Function")
        # feasibility
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
            pcm = ax_feasibility.pcolormesh(
                x_mesh[0].numpy(),
                x_mesh[1].numpy(),
                predictions["feasibility"].reshape(n_grid, n_grid),
            )
            divider = make_axes_locatable(ax_feasibility)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(pcm, cax=cax)
            cbar.set_label("Feasibility")
            ax_feasibility.set_title("Feasibility")
        else:
            if ncols == 3 and show_acquisition:
                ax[len(output_names), 2].axis("off")
    fig.tight_layout()
    return fig, ax
