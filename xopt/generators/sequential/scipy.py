from typing import Any, Dict, List, Optional

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
        super().__init__("scipy.optimize.minimize requested a new point evaluation")


class ScipyGenerator(SequentialGenerator):
    """
        Sequential wrapper around ``scipy.optimize.minimize``.

        Integration model
        -----------------
        Xopt uses ask/tell semantics (one new point per ``step``), while scipy
        ``minimize`` expects direct access to objective evaluations. This class bridges
        the two by replaying previously-evaluated points from ``self.data``:

        1. Build a cache from existing Xopt observations.
        2. Run ``scipy.optimize.minimize`` with an objective function that first checks
             the cache.
        3. If scipy asks for a point not in cache, raise ``_RequestEvaluation`` to exit
             minimize early and return that point to Xopt.
        4. Xopt evaluates the point externally and appends it to ``self.data``.
        5. The next ``step`` repeats the process with the larger cache.

        Performance implications
        ------------------------
        - ``_build_cache`` is O(N) in number of past evaluations and is executed every
            ``step``. This overhead is typically small when objective evaluations are
            expensive, but can dominate for very cheap objectives.
        - ``minimize`` is restarted from ``x0`` each ``step`` and progresses by replaying
            cached points. This is robust and method-agnostic, but adds repeated optimizer
            bookkeeping work compared to a persistent in-memory scipy run.
        - Point keys are rounded (12 decimals) before cache lookup to avoid fragile
            floating-point equality checks. This improves replay stability across methods
            that revisit numerically-close points.

    """

    name = "scipy"
    supports_single_objective: bool = True

    method: str = Field(
        "Powell",
        description="Method name passed to scipy.optimize.minimize (e.g. 'Powell', 'Nelder-Mead').",
    )
    initial_point: Optional[Dict[str, float]] = None
    tol: Optional[float] = Field(
        1e-8, description="Termination tolerance passed to scipy.optimize.minimize"
    )
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Options dictionary passed directly to scipy.optimize.minimize",
    )
    scipy_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments passed to scipy.optimize.minimize.",
    )

    # Internal state
    _last_outcome: Optional[float] = None

    @field_validator("method")
    def validate_method(cls, v: str):
        value = v.strip()
        if not value:
            raise ValueError("method cannot be empty")
        return value

    @field_validator("initial_point")
    def validate_initial_point(cls, v: Optional[Dict[str, float]]):
        if v is None:
            return v
        if len(v) == 0:
            raise ValueError("initial_point cannot be an empty dictionary")
        return v

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

        # Build a deterministic replay table from prior Xopt observations.
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
            # Exit scipy early when a new point is requested so Xopt can evaluate it.
            raise _RequestEvaluation(x)

        minimize_kwargs = {
            "method": self.method,
            "bounds": bounds,
            "tol": self.tol,
            "options": self.options,
            **self.scipy_kwargs,
        }

        try:
            minimize(objective, self.x0, **minimize_kwargs)
        except _RequestEvaluation as req:
            inputs = dict(zip(self.vocs.variable_names, req.x.tolist()))
            if self.vocs.constants is not None:
                inputs.update(self.vocs.constants)
            return [inputs]
        except ValueError as ex:
            msg = str(ex).lower()
            if "unknown solver" in msg:
                raise RuntimeError(
                    f"scipy method '{self.method}' is not available in this environment."
                ) from ex
            raise

        # If scipy converges using only cached data, return the latest known point.
        inputs = self.data[self.vocs.variable_names].iloc[-1].to_dict()
        if self.vocs.constants is not None:
            inputs.update(self.vocs.constants)
        return [inputs]
