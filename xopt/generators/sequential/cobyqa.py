from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import Field, field_validator
from scipy.optimize import minimize

from xopt.generators.sequential.sequential_generator import SequentialGenerator
from xopt.vocs import get_objective_data, get_variable_data


class _RequestEvaluation(Exception):
    """Internal exception used to request a new external function evaluation."""

    def __init__(self, x: np.ndarray):
        self.x = np.array(x, dtype=float)
        super().__init__("COBYQA requested a new point evaluation")


class COBYQAGenerator(SequentialGenerator):
    """
    Sequential COBYQA generator based on ``scipy.optimize.minimize``.

    This implementation follows an ask/tell pattern by replaying known evaluations from
    ``self.data``. Whenever scipy requests a point that has not been evaluated yet, the
    generator returns that point so that Xopt can evaluate it externally.
    """

    name = "cobyqa"
    supports_single_objective: bool = True

    initial_point: Optional[Dict[str, float]] = None
    tol: Optional[float] = Field(
        None, description="Termination tolerance passed to scipy.optimize.minimize"
    )
    options: Dict = Field(
        default_factory=dict,
        description="Options dictionary passed directly to scipy.optimize.minimize",
    )

    # Internal state
    _last_outcome: Optional[float] = None

    @field_validator("initial_point")
    def validate_initial_point(cls, v: Optional[Dict[str, float]]):
        if v is None:
            return v
        if len(v) == 0:
            raise ValueError("initial_point cannot be an empty dictionary")
        return v

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def x0(self) -> np.ndarray:
        if self.initial_point is not None:
            return np.array([self.initial_point[k] for k in self.vocs.variable_names])
        return self._get_initial_point()[0]

    def _reset(self):
        self._last_outcome = None

    def _set_data(self, data: pd.DataFrame):
        self.data = data
        if len(data) > 0:
            objective_data = get_objective_data(self.vocs, data).to_numpy()[:, 0]
            self._last_outcome = float(objective_data[-1])

    def _add_data(self, new_data: pd.DataFrame):
        if len(new_data) == 0:
            return
        objective_data = get_objective_data(self.vocs, new_data).to_numpy()[:, 0]
        self._last_outcome = float(objective_data[-1])

    def _point_key(self, x: np.ndarray, decimals: int = 12) -> tuple:
        return tuple(np.round(np.array(x, dtype=float), decimals=decimals))

    def _build_cache(self) -> dict[tuple, float]:
        if self.data is None or len(self.data) == 0:
            return {}

        x_data = get_variable_data(self.vocs, self.data).to_numpy(dtype=float)
        y_data = get_objective_data(self.vocs, self.data).to_numpy()[:, 0]

        return {self._point_key(x): float(y) for x, y in zip(x_data, y_data)}

    def _generate(self, first_gen: bool = False) -> Optional[List[Dict[str, float]]]:
        cache = self._build_cache()
        mins, maxs = np.array(self.vocs.bounds).T
        bounds = list(zip(mins, maxs))

        def objective(x):
            key = self._point_key(x)
            if key in cache:
                return cache[key]
            raise _RequestEvaluation(x)

        try:
            minimize(
                objective,
                self.x0,
                method="cobyqa",
                bounds=bounds,
                tol=self.tol,
                options=self.options,
            )
        except _RequestEvaluation as req:
            inputs = dict(zip(self.vocs.variable_names, req.x.tolist()))
            if self.vocs.constants is not None:
                inputs.update(self.vocs.constants)
            return [inputs]
        except ValueError as ex:
            msg = str(ex).lower()
            if "unknown solver" in msg and "cobyqa" in msg:
                raise RuntimeError(
                    "scipy COBYQA is not available in this environment. "
                    "Install a scipy version that supports minimize(method='cobyqa')."
                ) from ex
            raise

        # If scipy converges using only cached data, return the latest known point.
        last_x = self.x0
        inputs = dict(zip(self.vocs.variable_names, np.array(last_x).tolist()))
        if self.vocs.constants is not None:
            inputs.update(self.vocs.constants)
        return [inputs]
