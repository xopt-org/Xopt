from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr, field_validator
from scipy.optimize import minimize

from xopt.generators.sequential.sequential_generator import SequentialGenerator
from xopt.vocs import get_objective_data, get_variable_data

BOUNDED_METHODS = [
    "Nelder-Mead",
    "Powell",
    "L-BFGS-B",
    "TNC",
    "SLSQP",
    "trust-constr",
    "COBYLA",
    "COBYQA",
]


class _StopSession(Exception):
    """Internal exception used to terminate the persistent minimize session."""


class ScipyGenerator(SequentialGenerator):
    """
    Sequential wrapper around ``scipy.optimize.minimize``.

    Integration model
    -----------------
    Xopt uses ask/tell semantics (one new point per ``step``), while scipy
    ``minimize`` expects direct access to objective evaluations. This class bridges
    the two by running one persistent ``minimize`` call in a worker thread:

    1. Build a cache from existing Xopt observations.
    2. Start ``scipy.optimize.minimize`` once with an objective function that first
         checks the cache.
    3. If scipy asks for an unseen point, send that point to Xopt and block until
         Xopt provides the external objective value.
    4. Continue the same scipy run from in-memory state until convergence.

    Performance implications
    ------------------------
    - ``_build_cache`` is O(N) in number of past evaluations and is executed when the
        persistent session starts.
    - The active scipy run is maintained in memory between ``step`` calls.
    - Point keys are rounded (12 decimals) before cache lookup to avoid fragile
        floating-point equality checks.

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
    _cache: Dict[tuple, float] = PrivateAttr(default_factory=dict)
    _cache_lock: Lock = PrivateAttr(default_factory=Lock)
    _request_queue: Queue = PrivateAttr(default_factory=Queue)
    _response_queue: Queue = PrivateAttr(default_factory=Queue)
    _stop_event: Event = PrivateAttr(default_factory=Event)
    _session_thread: Optional[Thread] = PrivateAttr(default=None)
    _session_exception: Optional[Exception] = PrivateAttr(default=None)
    _session_finished: bool = PrivateAttr(default=False)
    _stop_token: object = PrivateAttr(default_factory=object)

    _runtime_private_attr_names = {
        "_cache_lock",
        "_request_queue",
        "_response_queue",
        "_stop_event",
        "_session_thread",
        "_session_exception",
        "_session_finished",
        "_stop_token",
    }

    @field_validator("method")
    def validate_method(cls, v: str):
        """Ensure scipy method names are not empty after whitespace normalization."""
        value = v.strip()
        if value not in BOUNDED_METHODS:
            raise ValueError(
                f"scipy method '{value}' is not supported; choose one of {BOUNDED_METHODS}"
            )
        return value

    @field_validator("initial_point")
    def validate_initial_point(cls, v: Optional[Dict[str, float]]):
        """Ensure that ``initial_point`` is either omitted or contains coordinates."""
        if v is not None and len(v) == 0:
            raise ValueError("initial_point cannot be an empty dictionary")
        return v

    def __deepcopy__(self, memo):
        """Create a safe deep copy without sharing runtime thread/session objects."""
        copied = self.__class__.model_validate(self.model_dump())
        if self.data is not None:
            copied.data = self.data.copy(deep=True)
            copied._last_outcome = self._last_outcome
            copied._cache = copied._build_cache()
        return copied

    def __getstate__(self):
        """Return pickle state while removing non-picklable runtime session objects."""
        # Do not pickle active threading primitives. Runtime session will be rebuilt.
        self._stop_session()
        state = super().__getstate__()
        private = state.get("__pydantic_private__", {}) or {}

        for key in self._runtime_private_attr_names:
            private.pop(key, None)

        private["_cache"] = self._build_cache()
        state["__pydantic_private__"] = private
        return state

    def __setstate__(self, state):
        """Restore pickle state and rebuild transient runtime synchronization objects."""
        super().__setstate__(state)
        self._cache_lock = Lock()
        self._request_queue = Queue()
        self._response_queue = Queue()
        self._stop_event = Event()
        self._session_thread = None
        self._session_exception = None
        self._session_finished = False
        self._stop_token = object()

        self._cache = self._build_cache()
        if self.data is not None and len(self.data) > 0:
            objective_data = get_objective_data(self.vocs, self.data).to_numpy()[:, 0]
            self._last_outcome = float(objective_data[-1])

    @property
    def x0(self) -> np.ndarray:
        """Return the optimization start point from config or from the latest dataset row."""
        if self.initial_point is not None:
            return np.array([self.initial_point[k] for k in self.vocs.variable_names])
        return self._get_initial_point()[0]

    def _reset(self):
        """Reset active session state while keeping existing evaluated observations."""
        self._stop_session()
        self._last_outcome = None

    def _set_data(self, data: pd.DataFrame):
        """Replace the full dataset and refresh cache/session-dependent internal state."""
        self._stop_session()
        self.data = data
        if len(data) > 0:
            objective_data = get_objective_data(self.vocs, data).to_numpy()[:, 0]
            self._last_outcome = float(objective_data[-1])
        self._cache = self._build_cache()

    def _add_data(self, new_data: pd.DataFrame):
        """Ingest one new evaluation and optionally unblock a waiting worker objective call."""
        if len(new_data) == 0:
            return
        objective_data = get_objective_data(self.vocs, new_data).to_numpy()[:, 0]
        self._last_outcome = float(objective_data[-1])

        x_value = get_variable_data(self.vocs, new_data).to_numpy(dtype=float)[-1]
        with self._cache_lock:
            self._cache[self._point_key(x_value)] = self._last_outcome

        if self._session_thread is not None and self._session_thread.is_alive():
            self._response_queue.put(self._last_outcome)

    def _point_key(self, x: np.ndarray, decimals: int = 12) -> tuple:
        """Convert a floating-point vector to a rounded hashable cache key."""
        return tuple(np.round(np.array(x, dtype=float), decimals=decimals))

    def _build_cache(self) -> dict[tuple, float]:
        """Build objective cache from ``self.data`` keyed by rounded variable vectors."""
        if self.data is None or len(self.data) == 0:
            return {}

        # Build a deterministic replay table from prior Xopt observations.
        x_data = get_variable_data(self.vocs, self.data).to_numpy(dtype=float)
        y_data = get_objective_data(self.vocs, self.data).to_numpy()[:, 0]

        return {self._point_key(x): float(y) for x, y in zip(x_data, y_data)}

    def _map_value_error(self, ex: ValueError) -> Optional[RuntimeError]:
        """Map scipy ``ValueError`` messages to clearer runtime configuration errors."""
        msg = str(ex).lower()
        if "unknown solver" in msg:
            return RuntimeError(
                f"scipy method '{self.method}' is not available in this environment."
            )
        if "cannot handle bounds" in msg:
            return RuntimeError(
                f"scipy method '{self.method}' does not support bounds; choose a bounded method (e.g. 'Powell', 'L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr')."
            )
        return None

    def _raise_session_error_if_present(self):
        """Raise any worker exception in the caller thread, with friendly remapping."""
        if self._session_exception is None:
            return

        ex = self._session_exception
        self._session_exception = None
        if isinstance(ex, ValueError):
            mapped = self._map_value_error(ex)
            if mapped is not None:
                raise mapped from ex
        raise ex

    def _objective(self, x: np.ndarray) -> float:
        """Objective callback used by scipy.

        This method first serves values from cache. For uncached points, it sends
        the requested point to the main thread and blocks until the corresponding
        externally evaluated objective value is provided.
        """
        if self._stop_event.is_set():
            raise _StopSession()

        key = self._point_key(x)
        with self._cache_lock:
            if key in self._cache:
                return self._cache[key]

        self._request_queue.put(np.array(x, dtype=float))
        response = self._response_queue.get()

        if response is self._stop_token or self._stop_event.is_set():
            raise _StopSession()

        y_value = float(response)
        with self._cache_lock:
            self._cache[key] = y_value
        return y_value

    def _run_session(self):
        """Run one persistent ``scipy.optimize.minimize`` session in a worker thread."""
        mins, maxs = np.array(self.vocs.bounds).T
        bounds = list(zip(mins, maxs))

        minimize_kwargs = {
            "method": self.method,
            "bounds": bounds,
            "tol": self.tol,
            "options": self.options,
            **self.scipy_kwargs,
        }

        try:
            minimize(self._objective, self.x0, **minimize_kwargs)
        except _StopSession:
            pass
        except Exception as ex:
            self._session_exception = ex
        finally:
            self._session_finished = True

    def _start_session_if_needed(self):
        """Start the worker minimize session if one is not currently active."""
        if self._session_thread is not None and self._session_thread.is_alive():
            return

        self._stop_event.clear()
        self._session_exception = None
        self._session_finished = False
        self._request_queue = Queue()
        self._response_queue = Queue()
        self._cache = self._build_cache()

        self._session_thread = Thread(target=self._run_session, daemon=True)
        self._session_thread.start()

    def _stop_session(self):
        """Request worker shutdown and reset transient synchronization primitives."""
        self._stop_event.set()
        if self._session_thread is not None and self._session_thread.is_alive():
            self._response_queue.put(self._stop_token)
            self._session_thread.join(timeout=2.0)

        self._session_thread = None
        self._session_exception = None
        self._session_finished = False
        self._request_queue = Queue()
        self._response_queue = Queue()
        self._stop_event = Event()

    def _generate(self, first_gen: bool = False) -> Optional[List[Dict[str, float]]]:
        """Return the next candidate requested by the live scipy session.

        The ``first_gen`` argument is accepted to satisfy the sequential generator
        interface; candidate selection always follows the active persistent session.
        """
        self._start_session_if_needed()

        while True:
            self._raise_session_error_if_present()

            try:
                requested_x = self._request_queue.get(timeout=0.05)
                inputs = dict(zip(self.vocs.variable_names, requested_x.tolist()))
                if self.vocs.constants is not None:
                    inputs.update(self.vocs.constants)
                return [inputs]
            except Empty:
                if self._session_finished:
                    self._raise_session_error_if_present()
                    break

        # If scipy converges using only cached data, return the latest known point.
        if self.data is None or len(self.data) == 0:
            raise RuntimeError("scipy minimize converged without available data")

        inputs = self.data[self.vocs.variable_names].iloc[-1].to_dict()
        if self.vocs.constants is not None:
            inputs.update(self.vocs.constants)
        return [inputs]
