from typing import Dict, List

import pandas as pd
import torch
from botorch.models.transforms import Normalize

from xopt.vocs import VOCS


def get_training_data(
    input_names: List[str], outcome_name: str, data: pd.DataFrame
) -> (torch.Tensor, torch.Tensor):
    """
    Returns (train_X, train_Y, train_Yvar) for the output specified by name.
    If `<outcome_name>_var` is in the dataframe then this function will also return a
    tensor for the y variance, else it returns None.

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


def set_botorch_weights(weights, vocs: VOCS):
    """set weights to multiply xopt objectives for botorch objectives"""
    for idx, ele in enumerate(vocs.objective_names):
        if vocs.objectives[ele] == "MINIMIZE":
            weights[idx] = -1.0
        elif vocs.objectives[ele] == "MAXIMIZE":
            weights[idx] = 1.0

    return weights


def get_input_transform(input_names: List, input_bounds: Dict[str, List] = None):
    if input_bounds is None:
        bounds = None
    else:
        bounds = torch.vstack(
            [torch.tensor(input_bounds[name]) for name in input_names]
        ).T
    return Normalize(len(input_names), bounds=bounds)


def rectilinear_domain_union(A, B):
    """
    outputs a rectilinear domain that is the union of bounds A/B

    A.shape = (2,N)
    B.shape = (2,N)
    """
    assert A.shape[0] == 2
    assert A.shape == B.shape

    out_bounds = torch.clone(A)

    out_bounds[0, :] = torch.max(
        A[0, :],
        B[0, :],
    )
    out_bounds[1, :] = torch.min(
        A[1, :],
        B[1, :],
    )

    return out_bounds
