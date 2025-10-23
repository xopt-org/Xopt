import warnings
import re

import numpy as np
import pandas as pd

from xopt.errors import FeasibilityError
from gest_api.vocs import (
    VOCS,
    GreaterThanConstraint,
    LessThanConstraint,
    BoundsConstraint,
    MaximizeObjective,
)


def random_inputs(
    vocs: VOCS,
    n: int = None,
    custom_bounds: dict[str, list[float]] = None,
    include_constants: bool = True,
    seed: int = None,
) -> list[dict]:
    """
    Uniform sampling of the variables.

    Returns a dict of inputs.

    If include_constants, the vocs.constants are added to the dict.

    Parameters
    ----------
    n : int, optional
        Number of samples to generate. Defaults to None.
    custom_bounds : dict, optional
        Custom bounds for the variables. Defaults to None.
    include_constants : bool, optional
        Whether to include constants in the inputs. Defaults to True.
    seed : int, optional
        Seed for the random number generator. Defaults to None.

    Returns
    -------
    list[dict]
        A list of dictionaries containing the sampled inputs.
    """
    inputs = {}
    if seed is None:
        rng_sample_function = np.random.random
    else:
        rng = np.random.default_rng(seed=seed)
        rng_sample_function = rng.random

    bounds = clip_variable_bounds(vocs, custom_bounds)

    for key, val in bounds.items():  # No need to sort here
        a, b = val
        x = rng_sample_function(n)
        inputs[key] = x * a + (1 - x) * b

    # Constants
    if include_constants and vocs.constants is not None:
        inputs.update({name: ele.value for name, ele in vocs.constants.items()})

    if n is None:
        return [inputs]
    else:
        return pd.DataFrame(inputs).to_dict("records")


def grid_inputs(
    vocs: VOCS,
    n: int | dict[str, int],
    custom_bounds: dict = None,
    include_constants: bool = True,
) -> pd.DataFrame:
    """
    Generate a meshgrid of inputs.

    Parameters
    ----------
    n : Union[int, Dict[str, int]]
        Number of points to generate along each axis. If an integer is provided, the same number of points
        is used for all variables. If a dictionary is provided, it should have variable names as keys and
        the number of points as values.
    custom_bounds : dict, optional
        Custom bounds for the variables. If None, the default bounds from `vocs.variables` are used.
        The dictionary should have variable names as keys and a list of two values [min, max] as values.
    include_constants : bool, optional
        If True, include constant values from `vocs.constants` in the output DataFrame.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the generated meshgrid of inputs. Each column corresponds to a variable,
        and each row represents a point in the grid.

    Raises
    ------
    TypeError
        If `custom_bounds` is not a dictionary.
    ValueError
        If `custom_bounds` are not valid or are outside the domain of `vocs.variables`.

    Warns
    -----
    RuntimeWarning
        If `custom_bounds` are clipped by the bounds of `vocs.variables`.

    Notes
    -----
    The function generates a meshgrid of inputs based on the specified bounds. If `custom_bounds` are provided,
    they are validated and clipped to ensure they lie within the domain of `vocs.variables`. The resulting meshgrid
    is flattened and returned as a DataFrame. If `include_constants` is True, constant values from `vocs.constants`
    are added to the DataFrame.
    """
    bounds = clip_variable_bounds(vocs, custom_bounds)

    grid_axes = []
    for key, val in bounds.items():
        if isinstance(n, int):
            num_points = n
        elif isinstance(n, dict) and key in n:
            num_points = n[key]
        else:
            raise ValueError(f"Number of points for variable '{key}' not specified.")
        grid_axes.append(np.linspace(val[0], val[1], num_points))

    mesh = np.meshgrid(*grid_axes)
    inputs = {key: mesh[i].flatten() for i, key in enumerate(bounds.keys())}

    if include_constants and vocs.constants is not None:
        for key, const in vocs.constants.items():
            inputs[key] = np.full_like(next(iter(inputs.values())), const.value)

    return pd.DataFrame(inputs)


def convert_dataframe_to_inputs(
    vocs: VOCS, data: pd.DataFrame, include_constants: bool = True
) -> pd.DataFrame:
    """
    Extracts only inputs from a dataframe.
    This will add constants if `include_constants` is true.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe to extract inputs from.
    include_constants : bool, optional
        Whether to include constants in the inputs. Defaults to True.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the extracted inputs.
    """
    # make sure that the df keys only contain vocs variables
    if not set(vocs.variable_names) == set(data.keys()):
        raise ValueError("input dataframe column set must equal set of vocs variables")

    # only keep the variables
    inner_copy = data.copy()

    # append constants if requested
    if include_constants:
        constants = vocs.constants
        if constants is not None:
            for name, var in constants.items():
                inner_copy[name] = var.value

    return inner_copy


def convert_numpy_to_inputs(
    vocs: VOCS, inputs: np.ndarray, include_constants: bool = True
) -> pd.DataFrame:
    """
    Convert 2D numpy array to list of dicts (inputs) for evaluation.
    Assumes that the columns of the array match correspond to
    `sorted(vocs.variables.keys())`

    Parameters
    ----------
    inputs : np.ndarray
        The 2D numpy array to convert.
    include_constants : bool, optional
        Whether to include constants in the inputs. Defaults to True.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the converted inputs.
    """
    df = pd.DataFrame(inputs, columns=vocs.variable_names)
    return convert_dataframe_to_inputs(vocs, df, include_constants)


def get_variable_data(
    vocs,
    data: pd.DataFrame | list[dict],
    prefix: str = "variable_",
) -> pd.DataFrame:
    """
    Returns a dataframe containing variables according to `vocs.variables` in sorted order.

    Parameters
    ----------
    vocs: VOCS
        The variable-objective-constraint space (VOCS) defining the problem.
    data : Union[pd.DataFrame, List[Dict]]
        The data to be processed.
    prefix : str, optional
        Prefix added to column names. Defaults to "variable_".

    Returns
    -------
    pd.DataFrame
        The processed dataframe.
    """
    if not vocs.variables:
        return pd.DataFrame([])

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    def custom_sort_key(s):
        """
        Split each string into its alphabetical and numeric parts.
        If the string is purely alphabetical, it sorts normally.
        If it's alphanumeric, it sorts by the letter(s) first, then by the number.
        """
        match = re.match(r"([a-zA-Z]+)(\d+)?$", s)
        if match:
            alpha = match.group(1)
            num = int(match.group(2)) if match.group(2) else -1
            return (alpha, num)
        else:
            return (s, -1)

    # Pick out columns in right order
    variables = sorted(vocs.variables, key=custom_sort_key)
    vdata = data.loc[:, variables].copy()
    # Rename to add prefix
    vdata.rename({k: prefix + k for k in variables})
    return vdata


def get_objective_data(
    vocs,
    data: pd.DataFrame | list[dict],
    prefix: str = "objective_",
    return_raw: bool = False,
) -> pd.DataFrame:
    """
    Returns a dataframe containing objective data transformed according to
    `vocs.objectives` such that we always assume minimization.

    Parameters
    ----------
    data : Union[pd.DataFrame, List[Dict]]
        The data to be processed.
    prefix : str, optional
        Prefix added to column names. Defaults to "objective_".
    return_raw : bool, optional
        Whether to return raw objective data. Defaults to False.

    Returns
    -------
    pd.DataFrame
        The processed dataframe.
    """
    if not vocs.objectives:
        return pd.DataFrame([])

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    objectives_names = sorted(vocs.objectives.keys())

    if set(data.columns).issuperset(set(objectives_names)):
        # have all objectives, dont need to fill in missing ones
        weights = np.ones(len(objectives_names))
        for i, k in enumerate(objectives_names):
            operator = vocs.objectives[k].__class__.__name__
            if operator not in OBJECTIVE_WEIGHT:
                raise ValueError(f"Unknown objective operator: {operator}")

            weights[i] = 1.0 if return_raw else OBJECTIVE_WEIGHT[operator]

        oarr = data.loc[:, objectives_names].to_numpy(copy=True, dtype=float) * weights
        oarr[np.isnan(oarr)] = np.inf
        odata = pd.DataFrame(
            oarr, columns=[prefix + k for k in objectives_names], index=data.index
        )
    else:
        # have to do this way because of missing objectives, even if slow
        # TODO: pre-allocate 2D array
        length = data.shape[0]
        array_list = []
        for i, k in enumerate(objectives_names):
            if k not in data:
                array_list.append(np.full((length, 1), np.inf))
                continue
            operator = vocs.objectives[k].__class__.__name__
            if operator not in OBJECTIVE_WEIGHT:
                raise ValueError(f"Unknown objective operator: {operator}")

            weight = 1.0 if return_raw else OBJECTIVE_WEIGHT[operator]
            arr = data.loc[:, [k]].to_numpy(copy=True, dtype=float) * weight
            arr[np.isnan(arr)] = np.inf
            array_list.append(arr)

        odata = pd.DataFrame(
            np.hstack(array_list),
            columns=[prefix + k for k in objectives_names],
            index=data.index,
        )

    return odata


def get_constraint_data(
    vocs,
    data: pd.DataFrame | list[dict],
    prefix: str = "constraint_",
) -> pd.DataFrame:
    """
    Returns a dataframe containing constraint data transformed according to
    `vocs.constraints` such that values that satisfy each constraint are negative.

    Parameters
    ----------
    data : Union[pd.DataFrame, List[Dict]]
        The data to be processed.
    prefix : str, optional
        Prefix added to column names. Defaults to "constraint_".

    Returns
    -------
    pd.DataFrame
        The processed dataframe.
    """
    if not vocs.constraints:
        return pd.DataFrame([])

    data = pd.DataFrame(data)  # cast to dataframe

    cdata = pd.DataFrame(index=data.index)

    for k in sorted(list(vocs.constraints)):
        # Protect against missing data
        if k not in data:
            cdata[prefix + k] = np.inf
            continue

        x = data[k]
        op = vocs.constraints[k]

        if isinstance(op, GreaterThanConstraint):  # x > d -> x-d > 0
            cvalues = -(x - op.value)
        elif isinstance(op, LessThanConstraint):  # x < d -> d-x > 0
            cvalues = -(op.value - x)
        elif isinstance(op, BoundsConstraint):  # x in [a,b] -> x-a > 0 and b-x > 0
            raise NotImplementedError("BoundsConstraint not implemented")
        else:
            raise ValueError(f"Unknown constraint operator: {op}")

        cdata[prefix + k] = cvalues.fillna(np.inf)  # Protect against nans
    return cdata.astype(float)


def get_observable_data(
    vocs,
    data: pd.DataFrame | list[dict],
    prefix: str = "observable_",
) -> pd.DataFrame:
    """
    Returns a dataframe containing observable data.

    Parameters
    ----------
    data : Union[pd.DataFrame, List[Dict]]
        The data to be processed.
    prefix : str, optional
        Prefix added to column names. Defaults to "observable_".

    Returns
    -------
    pd.DataFrame
        The processed dataframe.
    """
    if not vocs.observables:
        return pd.DataFrame([])

    data = pd.DataFrame(data)  # cast to dataframe

    cdata = pd.DataFrame(index=data.index)

    for k in vocs.observables:
        # Protect against missing data
        if k not in data:
            cdata[prefix + k] = np.inf
            continue

        ovalues = data[k]
        cdata[prefix + k] = ovalues.fillna(np.inf)  # Protect against nans
    return cdata


def get_feasibility_data(
    vocs,
    data: pd.DataFrame | list[dict],
    prefix: str = "feasible_",
) -> pd.DataFrame:
    """
    Returns a dataframe containing booleans denoting if a constraint is satisfied or
    not. Returned dataframe also contains a column `feasible` which denotes if
    all constraints are satisfied.

    Parameters
    ----------
    vocs : VOCS
        The variable-objective-constraint space (VOCS) defining the problem.
    data : Union[pd.DataFrame, List[Dict]]
        The data to be processed.
    prefix : str, optional
        Prefix added to column names. Defaults to "feasible_".

    Returns
    -------
    pd.DataFrame
        The processed dataframe.
    """
    if not vocs.constraints:
        df = pd.DataFrame(index=data.index)
        df["feasible"] = True
        return df

    data = pd.DataFrame(data)
    c_prefix = "constraint_"
    cdata = get_constraint_data(vocs, data, prefix=c_prefix)
    fdata = pd.DataFrame()

    for k in sorted(list(vocs.constraints)):
        fdata[prefix + k] = cdata[c_prefix + k].astype(float) <= 0
    # if all row values are true, then the row is feasible
    fdata["feasible"] = fdata.all(axis=1)
    return fdata


def normalize_inputs(vocs, input_points: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize input data (transform data into the range [0,1]) based on the
    variable ranges defined in the VOCS.

    Parameters
    ----------
    input_points : pd.DataFrame
        A DataFrame containing input data to be normalized.

    Returns
    -------
    pd.DataFrame
        A DataFrame with input data in the range [0,1] corresponding to the
        specified variable ranges. Contains columns equal to the intersection
        between `input_points` and `vocs.variable_names`.

    Notes
    -----

    If the input DataFrame is empty or no variable information is available in
    the VOCS, an empty DataFrame is returned.

    """
    normed_data = {}
    for name in vocs.variable_names:
        if name in input_points.columns:
            width = vocs.variables[name].domain[1] - vocs.variables[name].domain[0]
            normed_data[name] = (
                input_points[name] - vocs.variables[name].domain[0]
            ) / width

    if len(normed_data):
        return pd.DataFrame(normed_data)
    else:
        return pd.DataFrame([])


def denormalize_inputs(vocs, input_points: pd.DataFrame) -> pd.DataFrame:
    """
    Denormalize input data (transform data from the range [0,1]) based on the
    variable ranges defined in the VOCS.

    Parameters
    ----------
    input_points : pd.DataFrame
        A DataFrame containing normalized input data in the range [0,1].

    Returns
    -------
    pd.DataFrame
        A DataFrame with denormalized input data corresponding to the
        specified variable ranges. Contains columns equal to the intersection
        between `input_points` and `vocs.variable_names`.

    Notes
    -----

    If the input DataFrame is empty or no variable information is available in
    the VOCS, an empty DataFrame is returned.

    """
    denormed_data = {}
    for name in vocs.variable_names:
        if name in input_points.columns:
            width = vocs.variables[name].domain[1] - vocs.variables[name].domain[0]
            denormed_data[name] = (
                input_points[name] * width + vocs.variables[name].domain[0]
            )

    if len(denormed_data):
        return pd.DataFrame(denormed_data)
    else:
        return pd.DataFrame([])


def validate_input_data(vocs, input_points: pd.DataFrame) -> None:
    """
    Validates input data. Raises an error if the input data does not satisfy
    requirements given by vocs.

    Parameters
    ----------
        input_points : DataFrame
            Input data to be validated.

    Returns
    -------
        None

    Raises
    ------
        ValueError: if input data does not satisfy requirements.
    """
    variable_data = input_points.loc[:, vocs.variable_names].values
    bounds = np.array(vocs.bounds).T

    is_out_of_bounds_lower = variable_data < bounds[0, :]
    is_out_of_bounds_upper = variable_data > bounds[1, :]
    bad_mask = np.logical_or(is_out_of_bounds_upper, is_out_of_bounds_lower)
    any_bad = bad_mask.any()

    if any_bad:
        raise ValueError(
            f"input points at indices {np.nonzero(bad_mask.any(axis=0))} are not valid"
        )


def extract_data(vocs, data: pd.DataFrame, return_raw=False, return_valid=False):
    """
    split dataframe into seperate dataframes for variables, objectives and
    constraints based on vocs - objective data is transformed based on
    `vocs.objectives` properties

    Parameters
    ----------
        data: DataFrame
            Dataframe to be split
        return_raw : bool, optional
            If True, return untransformed objective data
        return_valid : bool, optional
            If True, only return data that satisfies all of the contraint
            conditions.

    Returns
    -------
        variable_data : DataFrame
            Dataframe containing variable data
        objective_data : DataFrame
            Dataframe containing objective data
        constraint_data : DataFrame
            Dataframe containing constraint data
        observable_data : DataFrame
            Dataframe containing observable data
    """
    variable_data = get_variable_data(vocs, data, "")
    objective_data = get_objective_data(vocs, data, "", return_raw)
    constraint_data = get_constraint_data(vocs, data, "")
    observable_data = get_observable_data(vocs, data, "")

    if return_valid:
        feasible_status = get_feasibility_data(vocs, data)["feasible"]
        return (
            variable_data.loc[feasible_status, :],
            objective_data.loc[feasible_status, :],
            constraint_data.loc[feasible_status, :],
            observable_data.loc[feasible_status, :],
        )

    return variable_data, objective_data, constraint_data, observable_data


def select_best(vocs, data: pd.DataFrame, n: int = 1):
    """
    get the best value and point for a given data set based on vocs
    - does not work for multi-objective problems
    - data that violates any constraints is ignored

    Parameters
    ----------
        data: DataFrame
            Dataframe to select best point from
        n: int, optional
            Number of best points to return

    Returns
    -------
        index: index of best point
        value: value of best point
        params: input parameters that give the best point
    """
    if vocs.n_objectives != 1:
        raise NotImplementedError("cannot select best point when n_objectives is not 1")

    if data.empty:
        raise RuntimeError("cannot select best point if dataframe is empty")

    feasible_data = get_feasibility_data(vocs, data)
    if feasible_data.empty or (~feasible_data["feasible"]).all():
        raise FeasibilityError(
            "Cannot select best point if no points satisfy the given constraints. "
        )

    ascending_flag = {"MinimizeObjective": True, "MaximizeObjective": False}
    obj = vocs.objectives[vocs.objective_names[0]].__class__.__name__
    obj_name = vocs.objective_names[0]

    res = (
        data.loc[feasible_data["feasible"], :]
        .sort_values(obj_name, ascending=ascending_flag[obj])
        .loc[:, obj_name]
        .iloc[:n]
    )

    params = data.loc[res.index, vocs.variable_names].to_dict(orient="records")[0]

    return (
        res.index.to_numpy(copy=True, dtype=int),
        res.to_numpy(copy=True, dtype=float),
        params,
    )


def cumulative_optimum(vocs, data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the cumulative optimum for the given DataFrame.

    Parameters
    ----------
    data: DataFrame
        Data for which the cumulative optimum shall be calculated.

    Returns
    -------
    DataFrame
        Cumulative optimum for the given DataFrame.

    """
    if not vocs.objectives:
        raise RuntimeError("No objectives defined.")
    if data.empty:
        return pd.DataFrame()
    obj_name = vocs.objective_names[0]
    obj = vocs.objectives[obj_name]
    get_opt = np.nanmax if isinstance(obj, MaximizeObjective) else np.nanmin
    feasible = get_feasibility_data(vocs, data)["feasible"]
    feasible_obj_values = [
        data[obj_name].values[i] if feasible[i] else np.nan for i in range(len(data))
    ]
    cumulative_optimum = np.array(
        [get_opt(feasible_obj_values[: i + 1]) for i in range(len(data))]
    )
    return pd.DataFrame({f"best_{obj_name}": cumulative_optimum}, index=data.index)


# --------------------------------
# dataframe utilities

OBJECTIVE_WEIGHT = {"MinimizeObjective": 1.0, "MaximizeObjective": -1.0}


def validate_variable_bounds(variable_dict: dict[str, list[float]]):
    """
    Check to make sure that bounds for variables are specified correctly. Raises
    ValueError if anything is incorrect
    """

    for name, value in variable_dict.items():
        if not isinstance(value, list):
            raise ValueError(f"Bounds specified for `{name}` must be a list.")
        if not len(value) == 2:
            raise ValueError(
                f"Bounds specified for `{name}` must be a list of length 2."
            )
        if not value[1] > value[0]:
            raise ValueError(
                f"Bounds specified for `{name}` do not satisfy the "
                f"condition value[1] > value[0]."
            )


def clip_variable_bounds(
    vocs: VOCS, custom_bounds: dict[str, list[float]]
) -> dict[str, list[float]]:
    """
    Return new bounds as intersection of vocs and custom bounds
    """
    if custom_bounds is None:
        final_bounds = dict(zip(vocs.variable_names, np.array(vocs.bounds)))
    else:
        variable_bounds = dict(zip(vocs.variable_names, np.array(vocs.bounds)))

        if not isinstance(custom_bounds, dict):
            raise TypeError("`custom_bounds` must be a dict")

        try:
            validate_variable_bounds(custom_bounds)
        except ValueError:
            raise ValueError("specified `custom_bounds` not valid")

        vars_clipped_lb_list = []
        vars_clipped_ub_list = []

        final_bounds = {}
        for var, (lb, ub) in variable_bounds.items():
            if var in custom_bounds:
                clb = custom_bounds[var][0]
                cub = custom_bounds[var][1]
                if clb >= ub:
                    # we already checked that clb < cub, so this is always an error
                    raise ValueError(
                        f"specified `custom_bounds` for {var} is outside vocs domain"
                    )
                if clb >= lb:
                    flb = clb
                else:
                    vars_clipped_lb_list.append(var)
                    flb = lb
                if cub <= ub:
                    fub = cub
                else:
                    vars_clipped_ub_list.append(var)
                    fub = ub
                final_bounds[var] = [flb, fub]
            else:
                final_bounds[var] = [lb, ub]

        if vars_clipped_lb_list:
            warnings.warn(
                f"custom bounds lower value exceeded vocs: {vars_clipped_lb_list}",
                RuntimeWarning,
            )
        if vars_clipped_ub_list:
            warnings.warn(
                f"custom bounds upper value exceeded vocs: {vars_clipped_ub_list}",
                RuntimeWarning,
            )

    return final_bounds
