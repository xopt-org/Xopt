import torch
from matplotlib import pyplot as plt

from xopt.generators.bayesian.visualize import (
    _generate_input_mesh,
    _get_reference_point,
)


def visualize_virtual_objective(
    generator,
    variable_names: list[str] = None,
    idx: int = -1,
    reference_point: dict = None,
    show_samples: bool = True,
    n_grid: int = 50,
    n_samples: int = 100,
    kwargs: dict = None,
) -> tuple:
    """
    Displays BAX's virtual objective predictions computed from samples drawn
    from the GP model(s) of the observable(s).

    Parameters
    ----------
        generator : Generator
            Bayesian generator object.
        variable_names : List[str]
            The variables with respect to which the GP models are displayed (maximum
            of 2). Defaults to generator.vocs.variable_names.
        idx : int
            Index of the last sample to use. This also selects the point of reference in
            higher dimensions unless an explicit reference_point is given.
        reference_point : dict
            Reference point determining the value of variables in
            generator.vocs.variable_names, but not in variable_names (slice plots in
            higher dimensions). Defaults to last used sample.
        show_samples : bool, optional
            Whether samples are shown.
        n_grid : int, optional
            Number of grid points per dimension used to display the model predictions.
        n_samples : int, optional
            Number of virtual objective samples to evaluate for each point in the scan.
        kwargs : dict, optional
            Additional keyword arguments for evaluating the virtual objective.

        Returns:
        --------
            The matplotlib figure and axes objects.
    """
    vocs, data = generator.vocs, generator.data
    reference_point = _get_reference_point(reference_point, vocs, data, idx)
    # define output and variable names
    if variable_names is None:
        variable_names = vocs.variable_names
    dim_x = len(variable_names)
    if dim_x not in [1, 2]:
        raise ValueError(
            f"Visualization is only supported with respect to 1 or 2 variables, "
            f"not {dim_x}. Provide a compatible list of variable names to create "
            f"slice plots at higher dimensions."
        )

    # validate variable names
    invalid = [name not in getattr(vocs, "variable_names") for name in variable_names]
    if any(invalid):
        invalid_names = [
            variable_names[i] for i in range(len(variable_names)) if invalid[i]
        ]
        raise ValueError(
            f"Variable names {invalid_names} are not in generator.vocs.variable_names."
        )

    # validate reference point keys
    invalid = [
        name not in getattr(vocs, "variable_names") for name in [*reference_point]
    ]
    if any(invalid):
        invalid_names = [
            [*reference_point][i] for i in range(len([*reference_point])) if invalid[i]
        ]
        raise ValueError(
            f"reference_point contains keys {invalid_names}, "
            f"which are not in generator.vocs.variable_names."
        )

    x = _generate_input_mesh(vocs, variable_names, reference_point, n_grid)

    # verify model exists
    if generator.model is None:
        raise ValueError(
            "The generator.model doesn't exist, try calling generator.train_model()."
        )

    # subset bax observable models
    bax_model_ids = [
        generator.vocs.output_names.index(name)
        for name in generator.algorithm.observable_names_ordered
    ]
    bax_model = generator.model.subset_output(bax_model_ids)

    # get virtual objective (sample) values
    bounds = generator._get_optimization_bounds()
    kwargs = kwargs if kwargs else {}
    objective_values = generator.algorithm.evaluate_virtual_objective(
        bax_model, x, bounds, tkwargs=generator._tkwargs, n_samples=n_samples, **kwargs
    )

    # get sample stats
    objective_med = objective_values.nanmedian(dim=0)[0].flatten()
    objective_upper = torch.nanquantile(objective_values, q=0.975, dim=0).flatten()
    objective_lower = torch.nanquantile(objective_values, q=0.025, dim=0).flatten()
    objective_std = (objective_upper - objective_lower) / 4

    figsize = (4 * dim_x, 3.7)
    fig, ax = plt.subplots(
        nrows=1, ncols=dim_x, sharex=True, sharey=True, figsize=figsize
    )

    if dim_x == 1:
        # 1d line plot
        x_axis = x[:, vocs.variable_names.index(variable_names[0])].squeeze().numpy()
        ax.plot(x_axis, objective_med, color="C0", label="Median")
        ax.fill_between(
            x_axis,
            objective_lower,
            objective_upper,
            color="C0",
            alpha=0.5,
            label="95% C.I.",
        )
        ax.legend()
        ax.set_ylabel("Virtual Objective")
        ax.set_xlabel(variable_names[0])
    else:
        # 2d heatmaps
        for j in [0, 1]:
            ax_j = ax[j]
            ax_j.locator_params(axis="both", nbins=5)
            if j == 0:
                prediction = objective_med
                title = "Objective Median"
                cbar_label = "Objective Median"
            elif j == 1:
                prediction = objective_std
                title = "Objective SD"
                cbar_label = r"$\sigma\,$[Objective]"

            pcm = ax_j.pcolormesh(
                x[:, vocs.variable_names.index(variable_names[0])]
                .reshape(n_grid, n_grid)
                .numpy(),
                x[:, vocs.variable_names.index(variable_names[1])]
                .reshape(n_grid, n_grid)
                .numpy(),
                prediction.reshape(n_grid, n_grid),
                rasterized=True,
            )

            from mpl_toolkits.axes_grid1 import make_axes_locatable  # lazy import

            divider = make_axes_locatable(ax_j)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(pcm, cax=cax)
            ax_j.set_title(title)
            ax_j.set_xlabel(variable_names[0])
            ax_j.set_ylabel(variable_names[1])
            cbar.set_label(cbar_label)

    fig.tight_layout()
    return fig, ax
