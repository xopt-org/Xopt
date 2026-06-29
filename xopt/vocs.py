"""
Variable and Objective Constraint (VOCS) Utilities

This module provides utilities and helper functions for working with VOCS (Variables,
Objectives, and Constraints) objects defined in the
generator standard library [gest-api](https://github.com/campa-consortium/gest-api). VOCS defines
the optimization problem's variables, objectives, and constraints,
serving as the foundation for all optimization algorithms.

"""

import warnings
from typing import Iterable
import numpy as np
import pandas as pd

from xopt.errors import FeasibilityError
from gest_api.vocs import (
    VOCS,
    ContinuousVariable,
    GreaterThanConstraint,
    LessThanConstraint,
    BoundsConstraint,
    DiscreteVariable,
    MaximizeObjective,
)


class ContextualVariable(ContinuousVariable):
    """
    A variable that is not optimized over, but rather is observed and can be conditioned on.

    By default, contextual variables are unbounded. In contexts that require finite bounds,
    bounds are inferred from the currently available data.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("domain", [-float("inf"), float("inf")])
        super().__init__(**kwargs)


def resolve_contextual_variable_bounds(
    variable: ContextualVariable,
    data: pd.Series | None,
    variable_name: str,
    padding_fraction: float = 0.05,
    minimum_padding: float = 1e-8,
) -> tuple[float, float]:
    """
    Resolve contextual variable bounds.

    Explicit finite domains always take precedence. If the contextual domain is
    unbounded, bounds are inferred from data with optional padding.
    """
    lower = float(variable.domain[0])
    upper = float(variable.domain[1])

    if np.isfinite(lower) and np.isfinite(upper):
        return lower, upper

    if data is None:
        raise KeyError(
            f"contextual variable `{variable_name}` requires data to infer finite bounds"
        )

    lower = float(data.min())
    upper = float(data.max())
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(
            f"contextual variable `{variable_name}` has non-finite bounds in data"
        )

    width = upper - lower
    padding = max(padding_fraction * width, minimum_padding)
    return lower - padding, upper + padding


def get_variable_bounds(
    vocs: VOCS,
    data: pd.DataFrame | None = None,
    variable_names: list[str] | None = None,
) -> dict[str, tuple[float, float]]:
    """Return [min, max] bounds for variables.

    Contextual variable bounds are always resolved with explicit-or-inferred
    behavior: explicit finite domains take precedence; unbounded domains are
    inferred from provided data.
    """
    bounds = {}
    names = variable_names if variable_names is not None else vocs.variable_names
    for name in names:
        variable = vocs.variables[name]
        if isinstance(variable, DiscreteVariable):
            values = sorted(float(v) for v in variable.values)
            bounds[name] = (values[0], values[-1])
        elif isinstance(variable, ContextualVariable):
            series = data[name] if data is not None and name in data else None
            lower, upper = resolve_contextual_variable_bounds(variable, series, name)
            bounds[name] = (lower, upper)
        else:
            bounds[name] = (float(variable.domain[0]), float(variable.domain[1]))
    return bounds


def get_variable_bounds_array(
    vocs: VOCS,
    data: pd.DataFrame | None = None,
    variable_names: list[str] | None = None,
) -> np.ndarray:
    """Return bounds for all variables as a 2 x d numpy array."""
    bounds = get_variable_bounds(vocs, data=data, variable_names=variable_names)
    return np.array(list(bounds.values())).T


def has_discrete_variables(vocs: VOCS) -> bool:
    """Check if there are any discrete variables in the VOCS."""
    for name in vocs.variable_names:
        variable = vocs.variables[name]
        if isinstance(variable, DiscreteVariable):
            return True
    return False


def random_inputs(
    vocs: VOCS,
    n: int = None,
    custom_bounds: dict[str, list[float]] = None,
    include_constants: bool = True,
    seed: int = None,
) -> list[dict]:
    """
    Generates uniform random samples of the variables as specified by VOCS.

    Parameters
    ----------
    vocs : VOCS
        The variable-objective-constraint space (VOCS) defining the problem.
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
        rng_choice_function = np.random.choice
    else:
        rng = np.random.default_rng(seed=seed)
        rng_sample_function = rng.random
        rng_choice_function = rng.choice

    bounds = clip_variable_bounds(vocs, custom_bounds)

    for key, val in bounds.items():
        variable = vocs.variables[key]

        if isinstance(variable, DiscreteVariable):
            a, b = val
            allowed_values = sorted(float(v) for v in variable.values)
            selectable_values = [v for v in allowed_values if a <= v <= b]

            if len(selectable_values) == 0:
                raise ValueError(
                    f"no discrete values for '{key}' inside bounds [{a}, {b}]"
                )

            if n is None:
                inputs[key] = float(rng_choice_function(selectable_values))
            else:
                inputs[key] = rng_choice_function(selectable_values, size=n)

        else:
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
    vocs : VOCS
        The variable-objective-constraint space (VOCS) defining the problem.
    n : Union[int, Dict[str, int]]
        Number of points to generate along each axis for continuous variables. If an integer is provided,
        the same number of points is used for all continuous variables. If a dictionary is provided, it
        should have continuous variable names as keys and the number of points as values. Discrete
        variables always use their configured values (optionally filtered by custom bounds).
    custom_bounds : dict, optional
        Custom bounds for the variables. If None, the default bounds from `vocs.variables` are used.
        The dictionary should have variable names as keys and a list of two values [min, max] as values.
        For discrete variables, bounds are used to filter allowed values.
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
    they are validated and clipped to ensure they lie within the domain of `vocs.variables`.

    Continuous variables are sampled using linspace. Discrete variables are enumerated using their allowed
    values (sorted ascending), filtered by active bounds when applicable.

    The resulting meshgrid is flattened and returned as a DataFrame. If `include_constants` is True, constant
    values from `vocs.constants` are added to the DataFrame.
    """
    bounds = clip_variable_bounds(vocs, custom_bounds)

    grid_axes = []
    for key, val in bounds.items():
        variable = vocs.variables[key]

        if isinstance(variable, DiscreteVariable):
            if isinstance(n, dict) and key in n:
                warnings.warn(
                    f"ignoring requested grid count for discrete variable '{key}'",
                    RuntimeWarning,
                )

            lb, ub = val
            discrete_values = sorted(float(v) for v in variable.values)
            filtered_values = [v for v in discrete_values if lb <= v <= ub]

            if len(filtered_values) == 0:
                raise ValueError(
                    f"no discrete values for '{key}' inside bounds [{lb}, {ub}]"
                )

            grid_axes.append(np.array(filtered_values, dtype=float))
            continue

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
    vocs: VOCS
        The variable-objective-constraint space (VOCS) defining the problem.
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
    non_contextual_variable_names = [
        name
        for name in vocs.variable_names
        if not isinstance(vocs.variables[name], ContextualVariable)
    ]
    if not set(non_contextual_variable_names) == set(data.keys()):
        raise ValueError(
            "input dataframe column set must equal set of non-contextual vocs variables"
        )

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
    Assumes that the columns of the array correspond to the non-contextual
    variables in `vocs.variable_names` order (contextual variables excluded).
    The number of columns in `inputs` must equal the number of non-contextual
    variables in the VOCS.

    Parameters
    ----------
    vocs: VOCS
        The variable-objective-constraint space (VOCS) defining the problem.
    inputs : np.ndarray
        The 2D numpy array to convert. Must have shape (n, k) where k is the
        number of non-contextual variables in vocs.
    include_constants : bool, optional
        Whether to include constants in the inputs. Defaults to True.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the converted inputs.
    """
    non_contextual_variable_names = [
        name
        for name in vocs.variable_names
        if not isinstance(vocs.variables[name], ContextualVariable)
    ]
    df = pd.DataFrame(inputs, columns=non_contextual_variable_names)
    return convert_dataframe_to_inputs(vocs, df, include_constants)


def get_variable_data(
    vocs: VOCS,
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

    # Pick out columns in right order
    vdata = data.loc[:, vocs.variable_names].copy()
    # Rename to add prefix
    vdata.rename({k: prefix + k for k in vocs.variable_names})
    return vdata


def get_objective_data(
    vocs: VOCS,
    data: pd.DataFrame | list[dict],
    prefix: str = "objective_",
    return_raw: bool = False,
) -> pd.DataFrame:
    """
    Returns a dataframe containing objective data transformed according to
    `objectives` such that we always assume minimization.

    Parameters
    ----------
    vocs: VOCS
        The variable-objective-constraint space (VOCS) defining the problem.
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
    vocs: VOCS,
    data: pd.DataFrame | list[dict],
    prefix: str = "constraint_",
) -> pd.DataFrame:
    """
    Returns a dataframe containing constraint data transformed according to
    `vocs.constraints` such that values that satisfy each constraint are negative.

    Parameters
    ----------
    vocs: VOCS
        The variable-objective-constraint space (VOCS) defining the problem.
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
    vocs: VOCS,
    data: pd.DataFrame | list[dict],
    prefix: str = "observable_",
) -> pd.DataFrame:
    """
    Returns a dataframe containing observable data.

    Parameters
    ----------
    vocs: VOCS
        The variable-objective-constraint space (VOCS) defining the problem.
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
    vocs: VOCS,
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


def normalize_inputs(vocs: VOCS, input_points: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize input data (transform data into the range [0,1]) based on the
    variable ranges defined in the VOCS.

    Parameters
    ----------
    vocs: VOCS
        The variable-objective-constraint space (VOCS) defining the problem.
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
    present_variable_names = [
        name for name in vocs.variable_names if name in input_points.columns
    ]
    variable_bounds = get_variable_bounds(
        vocs,
        data=input_points,
        variable_names=present_variable_names,
    )
    for name in vocs.variable_names:
        if name in input_points.columns:
            lb, ub = variable_bounds[name]
            width = ub - lb
            if np.isclose(width, 0.0):
                normed_data[name] = input_points[name].astype(float) - lb
            else:
                normed_data[name] = (input_points[name] - lb) / width

    if len(normed_data):
        return pd.DataFrame(normed_data)
    else:
        return pd.DataFrame([])


def denormalize_inputs(vocs: VOCS, input_points: pd.DataFrame) -> pd.DataFrame:
    """
    Denormalize input data (transform data from the range [0,1]) based on the
    variable ranges defined in the VOCS.

    Parameters
    ----------
    vocs: VOCS
        The variable-objective-constraint space (VOCS) defining the problem.
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
    present_variable_names = [
        name for name in vocs.variable_names if name in input_points.columns
    ]
    variable_bounds = get_variable_bounds(
        vocs,
        data=input_points,
        variable_names=present_variable_names,
    )
    for name in vocs.variable_names:
        if name in input_points.columns:
            lb, ub = variable_bounds[name]
            width = ub - lb
            if np.isclose(width, 0.0):
                denormed_data[name] = input_points[name].astype(float) + lb
            else:
                denormed_data[name] = input_points[name] * width + lb

    if len(denormed_data):
        return pd.DataFrame(denormed_data)
    else:
        return pd.DataFrame([])


def validate_input_data(vocs: VOCS, input_points: pd.DataFrame) -> None:
    """
    Validates input data. Raises an error if the input data does not satisfy
    requirements given by vocs.

    Parameters
    ----------
    vocs: VOCS
        The variable-objective-constraint space (VOCS) defining the problem.
    input_points : DataFrame
        Input data to be validated.

    Returns
    -------
    None

    Raises
    ------
        ValueError: if input data does not satisfy requirements.
    """

    variable_data = input_points.loc[:, vocs.variable_names]
    bad_mask = np.zeros((len(variable_data), len(vocs.variable_names)), dtype=bool)

    for idx, name in enumerate(vocs.variable_names):
        variable = vocs.variables[name]
        values = variable_data[name].astype(float).to_numpy()

        if isinstance(variable, DiscreteVariable):
            allowed = np.array(sorted(float(v) for v in variable.values), dtype=float)
            is_allowed = np.isclose(
                values[:, None], allowed[None, :], rtol=0.0, atol=1e-12
            ).any(axis=1)
            bad_mask[:, idx] = ~is_allowed
        elif isinstance(variable, ContextualVariable):
            # Validate contextual variables only when they have explicit finite
            # domains and values are present.
            if np.isfinite(variable.domain[0]) and np.isfinite(variable.domain[1]):
                present = ~np.isnan(values)
                finite = np.isfinite(values)
                lb = float(variable.domain[0])
                ub = float(variable.domain[1])

                bad_present = np.logical_and(
                    present,
                    np.logical_or(~finite, np.logical_or(values < lb, values > ub)),
                )
                bad_mask[:, idx] = bad_present
            else:
                continue
        else:
            lb = float(variable.domain[0])
            ub = float(variable.domain[1])
            bad_mask[:, idx] = np.logical_or(values < lb, values > ub)

    if bad_mask.any():
        row_indices = np.nonzero(bad_mask.any(axis=1))[0].tolist()
        bad_variables = [
            vocs.variable_names[i] for i in np.nonzero(bad_mask.any(axis=0))[0]
        ]
        raise ValueError(
            f"input points at row indices {row_indices} are not valid for "
            f"variables {bad_variables}"
        )


def extract_data(vocs: VOCS, data: pd.DataFrame, return_raw=False, return_valid=False):
    """
    split dataframe into seperate dataframes for variables, objectives and
    constraints based on vocs - objective data is transformed based on
    `vocs.objectives` properties

    Parameters
    ----------
    vocs: VOCS
        The variable-objective-constraint space (VOCS) defining the problem.
    data: DataFrame
        Dataframe to be split
    return_raw : bool, optional
        If True, return untransformed objective data
    return_valid : bool, optional
        If True, only return data that satisfies all of the constraint
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


def select_best(vocs: VOCS, data: pd.DataFrame, n: int = 1):
    """
    get the best value and point for a given data set based on vocs
    - does not work for multi-objective problems
    - does not work for EXPLORE objectives
    - data that violates any constraints is ignored

    Parameters
    ----------
    vocs: VOCS
        The variable-objective-constraint space (VOCS) defining the problem.
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

    obj_name = vocs.objective_names[0]
    obj = vocs.objectives[obj_name].__class__.__name__
    if obj == "ExploreObjective":
        raise NotImplementedError("cannot select best point for EXPLORE objective")

    if data.empty:
        raise RuntimeError("cannot select best point if dataframe is empty")

    feasible_data = get_feasibility_data(vocs, data)
    if feasible_data.empty or (~feasible_data["feasible"]).all():
        raise FeasibilityError(
            "Cannot select best point if no points satisfy the given constraints. "
        )

    ascending_flag = {"MinimizeObjective": True, "MaximizeObjective": False}
    if obj not in ascending_flag:
        raise NotImplementedError(f"cannot select best point for objective type: {obj}")

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


def cumulative_optimum(vocs: VOCS, data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the cumulative optimum for the given DataFrame.

    Parameters
    ----------
    vocs: VOCS
        The variable-objective-constraint space (VOCS) defining the problem.
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

    Parameters
    ----------
    variable_dict : dict[str, list[float]]
        Dictionary of variable bounds to validate

    Raises
    ------
    ValueError
        If bounds are not specified correctly

    Returns
    -------
    None
    """

    for name, value in variable_dict.items():
        if not isinstance(value, Iterable):
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
    vocs: VOCS, custom_bounds: dict[str, list[float]] | None = None
) -> dict[str, tuple[float, float]]:
    """
    Return new bounds as intersection of bounds provided by VOCS and custom bounds. 
    Note that this function only considers non-contextual variables, as 
    contextual variable bounds are not optimized over.

    Parameters
    ----------
    vocs : VOCS
        The variable-objective-constraint space (VOCS) defining the problem.
    custom_bounds : dict[str, list[float]]
        Custom bounds for the variables.

    Returns
    -------
    dict[str, list[float]]
        The final bounds after clipping custom bounds with vocs bounds.
    """
    active_variable_names = [
        name
        for name in vocs.variable_names
        if not isinstance(vocs.variables[name], ContextualVariable)
    ]

    if custom_bounds is None:
        final_bounds = get_variable_bounds(vocs, variable_names=active_variable_names)
    elif not isinstance(custom_bounds, dict):
        raise TypeError("specified `custom_bounds` must be a dict")
    else:
        variable_bounds = get_variable_bounds(vocs, variable_names=active_variable_names)

        try:
            validate_variable_bounds(custom_bounds)
        except ValueError:
            raise ValueError("specified `custom_bounds` not valid")

        vars_clipped_lb_list: list[str] = []
        vars_clipped_ub_list: list[str] = []

        final_bounds: dict[str, tuple[float, float]] = {}
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
                final_bounds[var] = (flb, fub)
            else:
                final_bounds[var] = (lb, ub)

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
