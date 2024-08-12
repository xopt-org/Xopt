from typing import List

import numpy as np
import pandas as pd
import torch

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
    assert A.shape == (2, A.shape[1]), "A should have shape (2, N)"
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


def validate_turbo_controller_base(value, available_controller_types, info):
    if isinstance(value, TurboController):
        valid_type = False
        for _, controller_type in available_controller_types.items():
            if isinstance(value, controller_type):
                valid_type = True

        if not valid_type:
            raise ValueError(
                f"turbo controller of type {type(value)} "
                f"not allowed for this generator"
            )

    elif isinstance(value, str):
        # create turbo controller from string input
        if value in available_controller_types:
            value = available_controller_types[value](info.data["vocs"])
        else:
            raise ValueError(
                f"{value} not found, available values are "
                f"{available_controller_types.keys()}"
            )
    elif isinstance(value, dict):
        # create turbo controller from dict input
        if "name" not in value:
            raise ValueError("turbo input dict needs to have a `name` attribute")
        name = value.pop("name")
        if name in available_controller_types:
            # pop unnecessary elements
            for ele in ["dim"]:
                value.pop(ele, None)

            value = available_controller_types[name](vocs=info.data["vocs"], **value)
        else:
            raise ValueError(
                f"{value} not found, available values are "
                f"{available_controller_types.keys()}"
            )
    return value
