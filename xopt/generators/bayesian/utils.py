from typing import List

import numpy as np
import pandas as pd
import torch

from xopt.vocs import VOCS


def get_training_data(
    input_names: List[str], outcome_name: str, data: pd.DataFrame
) -> (torch.Tensor, torch.Tensor):
    """Returns (train_X, train_Y) for the output specified by name."""
    input_data = data[input_names]
    outcome_data = data[outcome_name]

    # handle the case where multiple data points has been passed to the input dataframe
    expanded_input_data = []
    expanded_outcome_data = []
    for i, ele in enumerate(outcome_data):
        if isinstance(ele, list) or isinstance(ele, np.ndarray):
            expanded_outcome_data += ele
            expanded_input_data += [input_data.iloc[i]] * len(ele)
        else:
            expanded_outcome_data += [ele]
            expanded_input_data += [input_data.iloc[i]]

    input_data = pd.concat(expanded_input_data, ignore_index=True, axis=1).T
    outcome_data = pd.DataFrame(expanded_outcome_data)

    # cannot use any rows where any variable values are nans
    non_nans = ~input_data.isnull().T.any()
    input_data = input_data[non_nans]
    outcome_data = outcome_data[non_nans]

    non_nans_output = list(~outcome_data.isnull().T.iloc[0])
    train_X = torch.tensor(input_data[non_nans_output].to_numpy())
    train_Y = torch.tensor(outcome_data[non_nans_output].to_numpy())
    return train_X, train_Y


def set_botorch_weights(weights, vocs: VOCS):
    """set weights to multiply xopt objectives for botorch objectives"""
    for idx, ele in enumerate(vocs.objective_names):
        if vocs.objectives[ele] == "MINIMIZE":
            weights[idx] = -1.0
        elif vocs.objectives[ele] == "MAXIMIZE":
            weights[idx] = 1.0

    return weights


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
