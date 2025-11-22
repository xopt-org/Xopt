from contextlib import nullcontext
from copy import deepcopy
from typing import Any, List, cast

import gpytorch
import numpy as np
import pandas as pd
from pydantic import ValidationInfo
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.models import ModelListGP
from botorch.models.model import Model
from botorch.utils.multi_objective import is_non_dominated, Hypervolume

from xopt.generators.bayesian.turbo import TurboController
from xopt.vocs import VOCS


def get_training_data(
    input_names: List[str], outcome_name: str, data: pd.DataFrame
) -> (torch.Tensor, torch.Tensor):
    """
    Creates training data from input data frame.

    Parameters
    ----------
    input_names : List[str]
        List of input feature names.

    outcome_name : str
        Name of the outcome variable.

    data : pd.DataFrame
        DataFrame containing input and outcome data.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple containing training input tensor (train_X), training outcome tensor (
        train_Y), and training outcome variance tensor (train_Yvar).

    Notes
    -----

    The function handles NaN values, removing rows with NaN values in any of the
    input variables.

    If the DataFrame contains a column named `<outcome_name>_var`, the function
    returns a tensor for the outcome variance (train_Yvar); otherwise, train_Yvar is
    None.

    """

    input_data = data[input_names]
    outcome_data = data[outcome_name]

    # cannot use any rows where any variable values are nans
    non_nans = ~input_data.isnull().T.any()
    input_data = input_data[non_nans]
    outcome_data = outcome_data[non_nans]

    train_X = torch.tensor(input_data[~outcome_data.isnull()].to_numpy(dtype="double"))
    train_Y = torch.tensor(
        outcome_data[~outcome_data.isnull()].to_numpy(dtype="double")
    ).unsqueeze(-1)

    train_Yvar = None
    if f"{outcome_name}_var" in data:
        variance_data = data[f"{outcome_name}_var"][non_nans]
        train_Yvar = torch.tensor(
            variance_data[~outcome_data.isnull()].to_numpy(dtype="double")
        ).unsqueeze(-1)

    return train_X, train_Y, train_Yvar


def set_botorch_weights(vocs: VOCS):
    """set weights to multiply xopt objectives or observables for botorch objectives"""
    output_names = vocs.output_names

    weights = torch.zeros(len(output_names), dtype=torch.double)

    if vocs.n_objectives > 0:
        # if objectives exist this is an optimization problem
        # set weights according to the index of the models -- corresponds to the
        # ordering of output names
        for objective_name in vocs.objective_names:
            if vocs.objectives[objective_name] == "MINIMIZE":
                weights[output_names.index(objective_name)] = -1.0
            elif vocs.objectives[objective_name] == "MAXIMIZE":
                weights[output_names.index(objective_name)] = 1.0
    if vocs.n_objectives == 0:
        # if no objectives exist this may be an exploration problem, weight each
        # observable by 1.0
        for observable_name in vocs.observables:
            weights[output_names.index(observable_name)] = 1.0

    return weights


def rectilinear_domain_union(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Calculate the union of two rectilinear domains represented by input bounds A and B.

    Parameters
    ----------
    A : torch.Tensor
        Input bounds for domain A. It should have shape (2, N) where N is the number
        of dimensions. The first row contains the lower bounds, and the second row
        contains the upper bounds.

    B : torch.Tensor
        Input bounds for domain B. It should have the same shape as A.

    Returns
    -------
    torch.Tensor
        Output bounds representing the rectilinear domain that is the union of A and B.

    Raises
    ------
    AssertionError
        If the shape of A is not (2, N) or if the shape of A and B are not the same.

    Notes
    -----

    - The function assumes that the input bounds represent a rectilinear domain in
    N-dimensional space. - The output bounds represent the rectilinear domain
    obtained by taking the union of the input domains. - The lower bounds of the
    output domain are computed as the element-wise maximum of the lower bounds of A
    and B. - The upper bounds of the output domain are computed as the element-wise
    minimum of the upper bounds of A and B.

    Examples
    --------
    >>> A = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    >>> B = torch.tensor([[0.5, 1.5], [2.5, 3.5]])
    >>> result = rectilinear_domain_union(A, B)
    >>> print(result)
    tensor([[0.5, 1.0],
            [2.5, 3.0]])
    """
    assert A.shape[0] == 2, "A should have shape (2, N)"
    assert A.shape == B.shape, (
        "Shapes of A and B should be the same, current shapes "
        f"are {A.shape} and {B.shape}"
    )

    out_bounds = torch.clone(A)

    out_bounds[0, :] = torch.max(A[0, :], B[0, :])
    out_bounds[1, :] = torch.min(A[1, :], B[1, :])

    return out_bounds


def interpolate_points(df, num_points=10):
    """
    Generates interpolated points between two points specified by a pandas DataFrame.

    Parameters
    ----------
    df: DataFrame
        with two rows representing the start and end points.
    num_points: int
        Number of points to generate between the start and end points.

    Returns
    -------
    result: DataFrame
        DataFrame with the interpolated points.
    """
    if df.shape[0] != 2:
        raise ValueError("Input DataFrame must have exactly two rows.")

    start_point = df.iloc[0]
    end_point = df.iloc[1]

    # Create an array of num_points equally spaced between 0 and 1
    interpolation_factors = np.linspace(0, 1, num_points + 1)

    # Interpolate each column independently
    interpolated_points = pd.DataFrame()
    for col in df.columns:
        interpolated_values = np.interp(
            interpolation_factors, [0, 1], [start_point[col], end_point[col]]
        )
        interpolated_points[col] = interpolated_values[1:]

    return interpolated_points


def validate_turbo_controller_base(
    value: Any,
    valid_controller_types: list[type[TurboController]],
    info: ValidationInfo,
):
    """Validate turbo controller input"""

    # get string names of available controller types
    controller_types = {
        controller.__name__: controller for controller in valid_controller_types
    }

    vocs = info.data.get("vocs", None)
    if vocs is None:
        raise ValueError("vocs must be provided to validate turbo controller")

    if isinstance(value, str):
        # handle old string input
        if value == "optimize":
            value = "OptimizeTurboController"
        elif value == "safety":
            value = "SafetyTurboController"

        # create turbo controller from string input
        if value in controller_types:
            value = controller_types[value](vocs=vocs)
        else:
            raise ValueError(
                f"{value} not found, available values are {controller_types.keys()}"
            )
    elif isinstance(value, dict):
        value = cast(dict[str, Any], value)
        value_copy = deepcopy(value)
        # create turbo controller from dict input
        if "name" not in value:
            raise ValueError("turbo input dict needs to have a `name` attribute")
        name = value_copy.pop("name")
        if name in controller_types:
            # pop unnecessary elements
            for ele in ["dim", "vocs"]:
                value_copy.pop(ele, None)

            value = controller_types[name](vocs=vocs, **value_copy)
        else:
            raise ValueError(
                f"{value} not found, available values are {controller_types.keys()}"
            )

    # check if turbo controller is compatabile with the generator
    for controller_type in valid_controller_types:
        if isinstance(value, controller_type):
            return value
    else:
        raise ValueError(
            f"Turbo controller of type {type(value)} not allowed for this generator. Valid types are {valid_controller_types}"
        )


class MeanVarModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output_dist = self.model(x)
        return output_dist.mean, output_dist.variance


class MeanVarModelWrapperPosterior(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output_dist = self.model.posterior(x)
        return output_dist.mean, output_dist.variance


def torch_trace_gp_model(
    model: Model,
    vocs: VOCS,
    tkwargs: dict,
    posterior: bool = True,
    grad: bool = False,
    batch_size: int = 1,
    verify: bool = False,
) -> torch.jit.ScriptModule:
    """
    Trace a GPyTorch model using torch.jit.trace. Note that resulting object will return mean and variance directly,
    NOT a multivariate normal.

    Parameters
    ----------
    model : Model
        The GPyTorch model to compile.
    vocs : VOCS
        VOCS
    tkwargs : dict
        The keyword arguments for the torch tensor.
    posterior : bool, optional
        If True, prime the model by using posterior method, otherwise call directly (this invokes gpytorch posterior).
    grad : bool, optional
        If True, use gradient context, otherwise use no gradient context.
    batch_size : int, optional
        The batch size for the input tensor for tracing, by default 1.
    verify : bool, optional
        If True, request that torch verify the trace by comparing to eager mode, by default False.
    """
    if isinstance(model, ModelListGP):
        raise ValueError(
            "ModelListGP is not supported for JIT tracing - use individual models"
        )
    rand_point = vocs.random_inputs()[0]
    rand_vec = torch.stack(
        [rand_point[k] * torch.ones(batch_size) for k in vocs.variable_names], dim=1
    )
    test_x = rand_vec.to(**tkwargs)
    # test_x_1 = test_x[:1,...]

    gradctx = nullcontext() if grad else torch.no_grad()
    model.eval()
    with gradctx, gpytorch.settings.fast_pred_var(), gpytorch.settings.trace_mode():
        if posterior:
            pred = model.posterior(test_x)
            traced_model = torch.jit.trace(
                MeanVarModelWrapperPosterior(model), test_x, check_trace=False
            )
            traced_model = torch.jit.optimize_for_inference(traced_model)
        else:
            pred = model(test_x)
            traced_model = torch.jit.trace(
                MeanVarModelWrapper(model), test_x, check_trace=False
            )
            traced_model = torch.jit.optimize_for_inference(traced_model)
        if verify:
            traced_mean, traced_var = traced_model(test_x)
            assert torch.allclose(pred.mean, traced_mean, rtol=0), (
                f"JIT traced mean != original {pred.mean=} {traced_mean=}"
            )
            assert torch.allclose(pred.variance, traced_var, rtol=0), (
                f"JIT traced variance != original: {pred.variance=} {traced_var=}"
            )

    return traced_model.to(**tkwargs)


def torch_compile_gp_model(
    model: Model,
    vocs: VOCS,
    tkwargs: dict,
    backend: str = "inductor",
    mode="default",
    posterior=True,
    grad=False,
):
    """
    Compile a GPyTorch model using torch.compile, returning a compiled module

    Parameters
    ----------
    model : Model
        The GPyTorch model to compile.
    vocs : VOCS
        VOCS
    tkwargs : dict
        The keyword arguments for the torch tensor.
    backend : str, optional
        The backend for torch.compile, by default "inductor".
    mode : str, optional
        The mode for torch.compile, by default "default".
    posterior : bool, optional
        If True, prime the model by using posterior method, otherwise call directly (this invokes gpytorch posterior).
    grad : bool, optional
        If True, use gradient context, otherwise use no gradient context.
    """
    if isinstance(model, ModelListGP):
        raise ValueError("ModelListGP is not supported - use individual models")
    rand_point = vocs.random_inputs()[0]
    rand_vec = torch.stack(
        [rand_point[k] * torch.ones(1) for k in vocs.variable_names], dim=1
    )
    test_x = rand_vec.to(**tkwargs)

    gradctx = nullcontext if grad else torch.no_grad()
    # TODO: check if gpytorch trace mode faster
    with gradctx, gpytorch.settings.fast_pred_var():
        model.eval()
        if posterior:
            pred = model.posterior(test_x)
            traced_model = torch.compile(
                model, backend=backend, mode=mode, dynamic=None
            )
            mvn = traced_model.posterior(test_x)
        else:
            pred = model(test_x)
            traced_model = torch.compile(
                model, backend=backend, mode=mode, dynamic=None
            )
            mvn = traced_model(test_x)
        traced_mean, traced_var = mvn.mean, mvn.variance
        assert torch.allclose(pred.mean, traced_mean, rtol=0), (
            f"Compiled mean != original {pred.mean=} {traced_mean=}"
        )
        assert torch.allclose(pred.variance, traced_var, rtol=0), (
            f"Compiled variance != original: {pred.variance=} {traced_var=}"
        )

    return traced_model


def torch_trace_acqf(
    acq: AcquisitionFunction, vocs: VOCS, tkwargs: dict
) -> torch.jit.ScriptModule:
    """
    Trace an acquisition function using torch.jit.trace.

    Parameters
    ----------
    acq : AcquisitionFunction
        The acquisition function to trace.
    vocs : VOCS
        VOCS
    tkwargs : dict
        The keyword arguments for the torch tensor.
    """
    # Note that this is very fragile for when we mix q=1 and q>1 because tensors ndims changes
    rand_point = vocs.random_inputs()[0]
    rand_vec = torch.stack(
        [rand_point[k] * torch.ones(1) for k in vocs.variable_names], dim=1
    )
    test_x = rand_vec.to(**tkwargs)
    test_x = test_x.unsqueeze(-2)
    with gpytorch.settings.fast_pred_var(), gpytorch.settings.trace_mode():
        # Need dummy evaluation to set caches
        acq(test_x.clone().detach())
        saqcf = torch.jit.trace(
            acq,
            example_inputs=test_x.clone().detach(),
            check_trace=True,
            check_tolerance=1e-8,
        )
    return saqcf


def torch_compile_acqf(
    acq: AcquisitionFunction,
    vocs: VOCS,
    tkwargs: dict,
    backend: str = "inductor",
    mode="default",
    verify: bool = True,
):
    """
    Compile an acquisition function using torch.compile.

    Parameters
    ----------
    acq : AcquisitionFunction
        The acquisition function to compile.
    vocs : VOCS
        VOCS
    tkwargs : dict
        The keyword arguments for the torch tensor.
    backend : str, optional
        The backend for torch.compile, by default "inductor".
    mode : str, optional
        The mode for torch.compile, by default "default".
    verify : bool, optional
        If True, do the verification vs eager mode.
    """
    # TODO: check if trace mode better
    # NOTE: is verify is False, you need to ensure tensors are copied before calling
    # or RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run
    with gpytorch.settings.fast_pred_var(), gpytorch.settings.trace_mode():
        # assume that only a few shapes will happen - batch=1 and batch=nsamples
        saqcf = torch.compile(acq, backend=backend, mode=mode, dynamic=False)
        if verify:
            rand_point = vocs.random_inputs()[0]
            rand_vec = torch.stack(
                [rand_point[k] * torch.ones(1) for k in vocs.variable_names], dim=1
            )
            test_x = rand_vec
            test_x = test_x.unsqueeze(-2).to(**tkwargs)  # 1 x 1 x d
            acq_value = acq(test_x.clone().detach())
            sacq_value = saqcf(test_x.clone().detach())
            assert torch.allclose(acq_value, sacq_value, rtol=1e-10), (
                f"Compiled acquisition != original {acq_value=} {sacq_value=}"
            )
    return saqcf


def compute_hypervolume_and_pf(
    X: torch.Tensor,
    Y: torch.Tensor,
    reference_point: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, float]:
    """
    Compute the hypervolume and pareto front
    given a set of points assuming maximization.

    Parameters
    ----------
    X : torch.Tensor
        The input points.
    Y : torch.Tensor
        The objective values of the points.
    reference_point : torch.Tensor
        The reference point for hypervolume calculation.

    Returns
    -------
    pareto_front_X : torch.Tensor
        The points on the Pareto front. Returns None if no pareto front exists.
    pareto_front_Y : torch.Tensor
        The objective values of the points on the Pareto front. Returns None if no pareto front exists.
    pareto_mask : torch.Tensor
        A boolean mask indicating which points are on the Pareto front.
        Returns None if no pareto front exists.
    hv_value : float
        The hypervolume value.
    """

    hv = Hypervolume(reference_point)
    if Y.shape[0] == 0:
        return None, None, None, 0.0

    # add the reference point to the objective values
    # add a dummy point to the X values
    X = torch.vstack((torch.zeros(1, X.shape[1], dtype=X.dtype), X))
    Y = torch.vstack((reference_point.unsqueeze(0), Y))

    pareto_mask = is_non_dominated(Y)

    # if the first point is in the pareto front then
    # none of the points dominate over the reference
    if pareto_mask[0]:
        return None, None, None, 0.0

    # get pareto front points
    pareto_front_X = X[pareto_mask]
    pareto_front_Y = Y[pareto_mask]
    hv_value = hv.compute(Y[pareto_mask].cpu())

    return pareto_front_X, pareto_front_Y, pareto_mask, hv_value
