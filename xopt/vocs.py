import warnings
from enum import Enum
from typing import Any, Iterable, cast

import numpy as np
import pandas as pd
import yaml
from pydantic import ConfigDict, Field, field_validator
from xopt.errors import FeasibilityError

from xopt.pydantic import XoptBaseModel


# Enums for objectives and constraints
class ObjectiveEnum(str, Enum):
    MINIMIZE = "MINIMIZE"
    MAXIMIZE = "MAXIMIZE"

    # Allow any case
    @classmethod
    def _missing_(cls, value: object):
        if value is None:
            raise ValueError("ObjectiveEnum value cannot be None")
        try:
            sval = str(value)
        except Exception:
            raise ValueError(f"ObjectiveEnum value must be a string, got {type(value)}")
        for member in cls:
            if member.name.lower() == sval.lower():
                return member
        raise ValueError(f"Unknown ObjectiveEnum value: {value}")


class ConstraintEnum(str, Enum):
    LESS_THAN = "LESS_THAN"
    GREATER_THAN = "GREATER_THAN"

    # Allow any case
    @classmethod
    def _missing_(cls, value: object):
        if value is None:
            raise ValueError("ConstraintEnum value cannot be None")
        try:
            sval = str(value)
        except Exception:
            raise ValueError(
                f"ConstraintEnum value must be a string, got {type(value)}"
            )
        for member in cls:
            if member.name.lower() == sval.lower():
                return member
        raise ValueError(f"Unknown ConstraintEnum value: {value}")


class VOCS(XoptBaseModel):
    """
    Variables, Objectives, Constraints, and other Settings (VOCS) data structure
    to describe optimization problems.

    Attributes
    ----------
    variables : Dict[str, conlist(float, min_length=2, max_length=2)]
        Input variable names with a list of minimum and maximum values.
    constraints : Dict[str, conlist(Union[float, ConstraintEnum], min_length=2, max_length=2)]
        Constraint names with a list of constraint type and value.
    objectives : Dict[str, ObjectiveEnum]
        Objective names with type of objective.
    constants : Dict[str, Any]
        Constant names and values passed to evaluate function.
    observables : List[str]
        Observation names tracked alongside objectives and constraints.

    Methods
    -------
    from_yaml(cls, yaml_text: str) -> 'VOCS'
        Create a VOCS object from a YAML string.
    as_yaml(self) -> str
        Convert the VOCS object to a YAML string.
    random_inputs(self, n: int = None, custom_bounds: dict = None, include_constants: bool = True, seed: int = None) -> list[dict]
        Uniform sampling of the variables.
    convert_dataframe_to_inputs(self, data: pd.DataFrame, include_constants: bool = True) -> pd.DataFrame
        Extracts only inputs from a dataframe.
    convert_numpy_to_inputs(self, inputs: np.ndarray, include_constants: bool = True) -> pd.DataFrame
        Convert 2D numpy array to list of dicts (inputs) for evaluation.
    variable_data(self, data: Union[pd.DataFrame, List[Dict], List[Dict]], prefix: str = "variable_") -> pd.DataFrame
        Returns a dataframe containing variables according to `vocs.variables` in sorted order.
    objective_data(self, data: Union[pd.DataFrame, List[Dict], List[Dict]], prefix: str = "objective_", return_raw: bool = False) -> pd.DataFrame
        Returns a dataframe containing objective data transformed according to `vocs.objectives`.
    constraint_data(self, data: Union[pd.DataFrame, List[Dict], List[Dict]], prefix: str = "constraint_") -> pd.DataFrame
        Returns a dataframe containing constraint data transformed according to `vocs.constraints`.
    observable_data(self, data: Union[pd.DataFrame, List[Dict], List[Dict]], prefix: str = "observable_") -> pd.DataFrame
        Returns a dataframe containing observable data.
    feasibility_data(self, data: Union[pd.DataFrame, List[Dict], List[Dict]], prefix: str = "feasible_") -> pd.DataFrame
        Returns a dataframe containing booleans denoting if a constraint is satisfied or not.
    normalize_inputs(self, input_points: pd.DataFrame) -> pd.DataFrame
        Normalize input data (transform data into the range [0,1]) based on the variable ranges defined in the VOCS.
    denormalize_inputs(self, input_points: pd.DataFrame) -> pd.DataFrame
        Denormalize input data (transform data from the range [0,1]) based on the variable ranges defined in the VOCS.
    validate_input_data(self, input_points: pd.DataFrame) -> None
        Validates input data. Raises an error if the input data does not satisfy requirements given by vocs.
    extract_data(self, data: pd.DataFrame, return_raw: bool = False, return_valid: bool = False) -> tuple
        Split dataframe into separate dataframes for variables, objectives and constraints based on vocs.
    select_best(self, data: pd.DataFrame, n: int = 1) -> tuple
        Get the best value and point for a given data set based on vocs.
    cumulative_optimum(self, data: pd.DataFrame) -> pd.DataFrame
        Returns the cumulative optimum for the given DataFrame.
    """

    variables: dict[str, tuple[float, float]] = Field(
        default={},
        description="input variable names with a list of minimum and maximum values",
    )
    constraints: dict[str, tuple[ConstraintEnum, float]] = Field(
        default={},
        description="constraint names with a list of constraint type and value",
    )
    objectives: dict[str, ObjectiveEnum] = Field(
        default={}, description="objective names with type of objective"
    )
    constants: dict[str, Any] = Field(
        default={}, description="constant names and values passed to evaluate function"
    )
    observables: list[str] = Field(
        default=[],
        description="observation names tracked alongside objectives and constraints",
    )

    model_config = ConfigDict(
        validate_assignment=True, use_enum_values=True, extra="forbid"
    )

    @field_validator("variables", mode="before")
    @classmethod
    def fix_variables(cls, value: Any) -> dict[str, tuple[float, float]]:
        if not isinstance(value, dict):
            raise ValueError("must be a dictionary")

        for key, val in value.items():
            if not isinstance(key, str):
                raise ValueError("variable keys must be strings")

            if not isinstance(val, Iterable):
                raise ValueError("constraint values must be iterable")

            if len(val) != 2:
                raise ValueError("variable bounds must have length 2")

            # convert lists to tuples
            if not isinstance(val, tuple):
                try:
                    _val = tuple(val)
                    value[key] = _val
                except Exception as e:
                    raise ValueError(
                        f"could not convert list to tuple for key '{key}'"
                    ) from e

            # convert elements to float
            try:
                _val = (float(val[0]), float(val[1]))
                value[key] = _val
            except Exception as e:
                raise ValueError(
                    f"could not convert {val} to tuple of floats for key '{key}'"
                ) from e

        _value = cast(dict[str, tuple[float, float]], value)

        return _value

    @field_validator("variables", mode="after")
    @classmethod
    def correct_bounds_specification(cls, v: dict[str, tuple[float, float]]):
        validate_variable_bounds(v)
        return v

    @field_validator("constraints", mode="before")
    @classmethod
    def fix_constraints(cls, value: Any) -> dict[str, tuple[ConstraintEnum, float]]:
        if not isinstance(value, dict):
            raise ValueError("must be a dictionary")

        for key, val in value.items():
            if not isinstance(key, str):
                raise ValueError("constraint keys must be strings")

            if not isinstance(val, Iterable):
                raise ValueError("constraint values must be iterable")

            if len(val) != 2:
                raise ValueError("constraint bounds must have length 2")

            # convert lists to tuples
            if not isinstance(val, tuple):
                try:
                    _val = tuple(val)
                    value[key] = _val
                except Exception as e:
                    raise ValueError(
                        f"could not convert list to tuple for key '{key}'"
                    ) from e

            # convert first element to constraint enum
            try:
                _val = ConstraintEnum(val[0])
                value[key] = (_val, val[1])
            except Exception:
                raise ValueError(f"unknown constraint type '{val[0]}' for key '{key}'")

            # convert second element to float
            try:
                _val = float(val[1])
                value[key] = (val[0], _val)
            except Exception as e:
                raise ValueError(
                    f"could not convert {val[1]} to float for key '{key}'"
                ) from e

        _value = cast(dict[str, tuple[ConstraintEnum, float]], value)

        return _value

    @field_validator("constraints", mode="after")
    @classmethod
    def correct_list_types(cls, v: dict[str, tuple[str, float]]):
        """make sure that constraint list types are correct"""
        for _, item in v.items():
            if not isinstance(item[0], str):
                raise ValueError(
                    "constraint specification list must have the first "
                    "element as a string`"
                )

            if not isinstance(item[1], float):
                raise ValueError(
                    "constraint specification list must have the second "
                    "element as a float"
                )

        return v

    @classmethod
    def from_yaml(cls, yaml_text: str) -> "VOCS":
        """
        Create a VOCS object from a YAML string.

        Parameters
        ----------
        yaml_text : str
            The YAML string to create the VOCS object from.

        Returns
        -------
        VOCS
            The created VOCS object.
        """
        loaded = yaml.safe_load(yaml_text)
        return cls(**loaded)

    def as_yaml(self) -> str:
        """
        Convert the VOCS object to a YAML string.

        Returns
        -------
        str
            The YAML string representation of the VOCS object.
        """
        return yaml.dump(self.model_dump(), default_flow_style=None, sort_keys=False)

    @property
    def bounds(self) -> np.ndarray:
        """
        Returns a bounds array (mins, maxs) of shape (2, n_variables).
        Arrays of lower and upper bounds can be extracted by:
            mins, maxs = vocs.bounds

        Returns
        -------
        np.ndarray
            The bounds array.
        """
        return np.array([v for _, v in sorted(self.variables.items())]).T

    @property
    def variable_names(self) -> list[str]:
        """Returns a sorted list of variable names"""
        return list(sorted(self.variables.keys()))

    @property
    def objective_names(self) -> list[str]:
        """Returns a sorted list of objective names"""
        return list(sorted(self.objectives.keys()))

    @property
    def constraint_names(self) -> list[str]:
        """Returns a sorted list of constraint names"""
        if self.constraints is None:
            return []
        return list(sorted(self.constraints.keys()))

    @property
    def observable_names(self) -> list[str]:
        """Returns a sorted list of observable names"""
        return sorted(self.observables)

    @property
    def output_names(self) -> list[str]:
        """
        Returns a list of expected output keys:
            (objectives + constraints + observables)
        Each sub-list is sorted.

        Returns
        -------
        List[str]
            The list of expected output keys.
        """
        full_list = self.objective_names
        for ele in self.constraint_names:
            if ele not in full_list:
                full_list += [ele]

        for ele in self.observable_names:
            if ele not in full_list:
                full_list += [ele]

        return full_list

    @property
    def constant_names(self) -> list[str]:
        """Returns a sorted list of constant names"""
        if self.constants is None:
            return []
        return list(sorted(self.constants.keys()))

    @property
    def all_names(self) -> list[str]:
        """Returns all vocs names (variables, constants, objectives, constraints)"""
        return self.variable_names + self.constant_names + self.output_names

    @property
    def n_variables(self) -> int:
        """Returns the number of variables"""
        return len(self.variables)

    @property
    def n_constants(self) -> int:
        """Returns the number of constants"""
        return len(self.constants)

    @property
    def n_inputs(self) -> int:
        """Returns the number of inputs (variables and constants)"""
        return self.n_variables + self.n_constants

    @property
    def n_objectives(self) -> int:
        """Returns the number of objectives"""
        return len(self.objectives)

    @property
    def n_constraints(self) -> int:
        """Returns the number of constraints"""
        return len(self.constraints)

    @property
    def n_observables(self) -> int:
        """Returns the number of observables"""
        return len(self.observables)

    @property
    def n_outputs(self) -> int:
        """
        Returns the number of outputs
            len(objectives + constraints + observables)

        Returns
        -------
        int
            The number of outputs.
        """
        return len(self.output_names)

    def random_inputs(
        self,
        n: int | None = None,
        custom_bounds: dict[str, list[float]] | None = None,
        include_constants: bool = True,
        seed: int | None = None,
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

        bounds = clip_variable_bounds(self, custom_bounds)

        for key, val in bounds.items():  # No need to sort here
            a, b = val
            x = rng_sample_function(n)
            inputs[key] = x * a + (1 - x) * b

        # Constants
        if include_constants and self.constants is not None:
            inputs.update(self.constants)

        if n is None:
            return [inputs]
        else:
            return pd.DataFrame(inputs).to_dict("records")

    def grid_inputs(
        self,
        n: int | dict[str, int],
        custom_bounds: dict[str, list[float]] | None = None,
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
            Custom bounds for the variables. If None, the default bounds from `self.variables` are used.
            The dictionary should have variable names as keys and a list of two values [min, max] as values.
        include_constants : bool, optional
            If True, include constant values from `self.constants` in the output DataFrame.

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
            If `custom_bounds` are not valid or are outside the domain of `self.variables`.

        Warns
        -----
        RuntimeWarning
            If `custom_bounds` are clipped by the bounds of `self.variables`.

        Notes
        -----
        The function generates a meshgrid of inputs based on the specified bounds. If `custom_bounds` are provided,
        they are validated and clipped to ensure they lie within the domain of `self.variables`. The resulting meshgrid
        is flattened and returned as a DataFrame. If `include_constants` is True, constant values from `self.constants`
        are added to the DataFrame.
        """
        bounds = clip_variable_bounds(self, custom_bounds)

        grid_axes = []
        for key, val in bounds.items():
            if isinstance(n, int):
                num_points = n
            elif isinstance(n, dict) and key in n:
                num_points = n[key]
            else:
                raise ValueError(
                    f"Number of points for variable '{key}' not specified."
                )
            grid_axes.append(np.linspace(val[0], val[1], num_points))

        mesh = np.meshgrid(*grid_axes)
        inputs = {key: mesh[i].flatten() for i, key in enumerate(bounds.keys())}

        if include_constants:
            for key, value in self.constants.items():
                inputs[key] = np.full_like(next(iter(inputs.values())), value)

        return pd.DataFrame(inputs)

    def convert_dataframe_to_inputs(
        self, data: pd.DataFrame, include_constants: bool = True
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
        if not set(self.variable_names) == set(data.keys()):
            raise ValueError(
                "input dataframe column set must equal set of vocs variables"
            )

        # only keep the variables
        inner_copy = data.copy()

        # append constants if requested
        if include_constants:
            constants = self.constants

            for name, val in constants.items():
                inner_copy[name] = val

        return inner_copy

    def convert_numpy_to_inputs(
        self, inputs: np.ndarray, include_constants: bool = True
    ) -> pd.DataFrame:
        """
        Convert 2D numpy array to list of dicts (inputs) for evaluation.
        Assumes that the columns of the array match correspond to
        `sorted(self.vocs.variables.keys())`

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
        df = pd.DataFrame(inputs, columns=self.variable_names)
        return self.convert_dataframe_to_inputs(df, include_constants)

    def variable_data(
        self,
        data: pd.DataFrame | list[dict[str, Any]],
        prefix: str = "variable_",
    ) -> pd.DataFrame:
        """
        Returns a dataframe containing variables according to `vocs.variables` in sorted order.

        Parameters
        ----------
        data : Union[pd.DataFrame, List[Dict]]
            The data to be processed.
        prefix : str, optional
            Prefix added to column names. Defaults to "variable_".

        Returns
        -------
        pd.DataFrame
            The processed dataframe.
        """
        return form_variable_data(self.variables, data, prefix=prefix)

    def objective_data(
        self,
        data: pd.DataFrame | list[dict[str, Any]],
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
        return form_objective_data(self.objectives, data, prefix, return_raw)

    def constraint_data(
        self,
        data: pd.DataFrame | list[dict[str, Any]],
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
        return form_constraint_data(self.constraints, data, prefix)

    def observable_data(
        self,
        data: pd.DataFrame | list[dict[str, Any]],
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
        return form_observable_data(self.observable_names, data, prefix)

    def feasibility_data(
        self,
        data: pd.DataFrame | list[dict[str, Any]],
        prefix: str = "feasible_",
    ) -> pd.DataFrame:
        """
        Returns a dataframe containing booleans denoting if a constraint is satisfied or
        not. Returned dataframe also contains a column `feasible` which denotes if
        all constraints are satisfied.

        Parameters
        ----------
        data : Union[pd.DataFrame, List[Dict]]
            The data to be processed.
        prefix : str, optional
            Prefix added to column names. Defaults to "feasible_".

        Returns
        -------
        pd.DataFrame
            The processed dataframe.
        """
        return form_feasibility_data(self.constraints, data, prefix)

    def normalize_inputs(self, input_points: pd.DataFrame) -> pd.DataFrame:
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
        for name in self.variable_names:
            if name in input_points.columns:
                width = self.variables[name][1] - self.variables[name][0]
                normed_data[name] = (
                    input_points[name] - self.variables[name][0]
                ) / width

        if len(normed_data):
            return pd.DataFrame(normed_data)
        else:
            return pd.DataFrame([])

    def denormalize_inputs(self, input_points: pd.DataFrame) -> pd.DataFrame:
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
        for name in self.variable_names:
            if name in input_points.columns:
                width = self.variables[name][1] - self.variables[name][0]
                denormed_data[name] = (
                    input_points[name] * width + self.variables[name][0]
                )

        if len(denormed_data):
            return pd.DataFrame(denormed_data)
        else:
            return pd.DataFrame([])

    def validate_input_data(self, input_points: pd.DataFrame) -> None:
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
        validate_input_data(self, input_points)

    def extract_data(self, data: pd.DataFrame, return_raw=False, return_valid=False):
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
        variable_data = self.variable_data(data, "")
        objective_data = self.objective_data(data, "", return_raw)
        constraint_data = self.constraint_data(data, "")
        observable_data = self.observable_data(data, "")

        if return_valid:
            feasible_status = self.feasibility_data(data)["feasible"]
            return (
                variable_data.loc[feasible_status, :],
                objective_data.loc[feasible_status, :],
                constraint_data.loc[feasible_status, :],
                observable_data.loc[feasible_status, :],
            )

        return variable_data, objective_data, constraint_data, observable_data

    def select_best(self, data: pd.DataFrame, n: int = 1):
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
        if self.n_objectives != 1:
            raise NotImplementedError(
                "cannot select best point when n_objectives is not 1"
            )

        if data.empty:
            raise RuntimeError("cannot select best point if dataframe is empty")

        feasible_data = self.feasibility_data(data)
        if feasible_data.empty or (~feasible_data["feasible"]).all():
            raise FeasibilityError(
                "Cannot select best point if no points satisfy the given constraints. "
            )

        ascending_flag = {"MINIMIZE": True, "MAXIMIZE": False}
        obj = self.objectives[self.objective_names[0]]
        obj_name = self.objective_names[0]

        res = (
            data.loc[feasible_data["feasible"], :]
            .sort_values(obj_name, ascending=ascending_flag[obj])
            .loc[:, obj_name]
            .iloc[:n]
        )

        params = data.loc[res.index, self.variable_names].to_dict(orient="records")[0]

        return (
            res.index.to_numpy(copy=True, dtype=int),
            res.to_numpy(copy=True, dtype=float),
            params,
        )

    def cumulative_optimum(self, data: pd.DataFrame) -> pd.DataFrame:
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
        if not self.objectives:
            raise RuntimeError("No objectives defined.")
        if data.empty:
            return pd.DataFrame()
        obj_name = self.objective_names[0]
        obj = self.objectives[obj_name]
        get_opt = np.nanmax if obj == "MAXIMIZE" else np.nanmin
        feasible = self.feasibility_data(data)["feasible"]
        feasible_obj_values = [
            data[obj_name].values[i] if feasible[i] else np.nan
            for i in range(len(data))
        ]
        cumulative_optimum = np.array(
            [get_opt(feasible_obj_values[: i + 1]) for i in range(len(data))]
        )
        return pd.DataFrame({f"best_{obj_name}": cumulative_optimum}, index=data.index)


# --------------------------------
# dataframe utilities

OBJECTIVE_WEIGHT = {"MINIMIZE": 1.0, "MAXIMIZE": -1.0}


def form_variable_data(
    variables: dict | pd.DataFrame, data, prefix="variable_"
) -> pd.DataFrame:
    """
    Use variables dict to form a dataframe.
    """
    if not variables:
        return pd.DataFrame([])

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # Pick out columns in right order
    variables = sorted(variables)
    vdata = data.loc[:, variables].copy()
    # Rename to add prefix
    vdata.rename({k: prefix + k for k in variables})
    return vdata


def form_objective_data(
    objectives: dict, data, prefix="objective_", return_raw: bool = False
) -> pd.DataFrame:
    """
    Use objective dict and data (dataframe) to generate objective data (dataframe)

    Weights are applied to convert all objectives into minimization form unless
    `return_raw` is True

    Returns a dataframe with the objective data intended to be minimized.

    Missing or nan values will be filled with: np.inf

    """
    if not objectives:
        return pd.DataFrame([])

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    objectives_names = sorted(objectives.keys())

    if set(data.columns).issuperset(set(objectives_names)):
        # have all objectives, dont need to fill in missing ones
        weights = np.ones(len(objectives_names))
        for i, k in enumerate(objectives_names):
            operator = objectives[k].upper()
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
            operator = objectives[k].upper()
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


def form_constraint_data(
    constraints: dict, data: pd.DataFrame, prefix="constraint_"
) -> pd.DataFrame:
    """
    Use constraint dict and data (dataframe) to generate constraint data (dataframe). A
    constraint is satisfied if the evaluation is < 0.

    Parameters
    ----------
        constraints: dict
            Dictionary of constraints
        data: DataFrame
            Dataframe with the data to be evaluated
        prefix: str, optional
            Prefix to use for the transformed data in the dataframe

    Returns a dataframe with the constraint data.

    Missing or nan values will be filled with: np.inf

    """
    if not constraints:
        return pd.DataFrame([])

    data = pd.DataFrame(data)  # cast to dataframe

    cdata = pd.DataFrame(index=data.index)

    for k in sorted(list(constraints)):
        # Protect against missing data
        if k not in data:
            cdata[prefix + k] = np.inf
            continue

        x = data[k].astype(float)
        op, d = constraints[k]
        op = op.upper()  # Allow any case

        if op == "GREATER_THAN":  # x > d -> x-d > 0
            cvalues = -(x - d)
        elif op == "LESS_THAN":  # x < d -> d-x > 0
            cvalues = -(d - x)
        else:
            raise ValueError(f"Unknown constraint operator: {op}")

        cdata[prefix + k] = cvalues.fillna(np.inf)  # Protect against nans
    return cdata.astype(float)


def form_observable_data(
    observables: list, data: pd.DataFrame, prefix="observable_"
) -> pd.DataFrame:
    """
    Use constraint dict and data (dataframe) to generate constraint data (dataframe). A
    constraint is satisfied if the evaluation is < 0.

    Parameters
    ----------
        observables: dict
            Dictionary of observables
        data: DataFrame
            Dataframe with the data to be evaluated
        prefix: str, optional
            Prefix to use for the transformed data in the dataframe

    Returns a dataframe with the constraint data.

    Missing or nan values will be filled with: np.inf

    """
    if not observables:
        return pd.DataFrame([])

    data = pd.DataFrame(data)  # cast to dataframe

    cdata = pd.DataFrame(index=data.index)

    for k in observables:
        # Protect against missing data
        if k not in data:
            cdata[prefix + k] = np.inf
            continue

        ovalues = data[k]
        cdata[prefix + k] = ovalues.fillna(np.inf)  # Protect against nans
    return cdata


def form_feasibility_data(constraints: dict, data, prefix="feasible_") -> pd.DataFrame:
    """
    Use constraint dict and data to identify feasible points in the dataset.

    Returns a dataframe with the feasible data.
    """
    if not constraints:
        df = pd.DataFrame(index=data.index)
        df["feasible"] = True
        return df

    data = pd.DataFrame(data)
    c_prefix = "constraint_"
    cdata = form_constraint_data(constraints, data, prefix=c_prefix)
    fdata = pd.DataFrame()

    for k in sorted(list(constraints)):
        fdata[prefix + k] = cdata[c_prefix + k].astype(float) <= 0
    # if all row values are true, then the row is feasible
    fdata["feasible"] = fdata.all(axis=1)
    return fdata


def validate_input_data(vocs: VOCS, data: pd.DataFrame) -> None:
    variable_data = data.loc[:, vocs.variable_names].values
    bounds = vocs.bounds  # type: ignore

    is_out_of_bounds_lower = variable_data < bounds[0, :]
    is_out_of_bounds_upper = variable_data > bounds[1, :]
    bad_mask = np.logical_or(is_out_of_bounds_upper, is_out_of_bounds_lower)
    any_bad = bad_mask.any()

    if any_bad:
        raise ValueError(
            f"input points at indices {np.nonzero(bad_mask.any(axis=0))} are not valid"
        )


def validate_variable_bounds(variable_dict: dict[str, tuple[float, float]]) -> None:
    """
    Check to make sure that bounds for variables are specified correctly. Raises
    ValueError if anything is incorrect
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
    Return new bounds as intersection of vocs and custom bounds
    """
    if custom_bounds is None:
        final_bounds = vocs.variables
    else:
        variable_bounds = vocs.variables

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
