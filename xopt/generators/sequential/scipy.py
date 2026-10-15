import copy
import warnings
from functools import partial
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr, field_validator
from scipy.optimize import OptimizeResult, minimize

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
    _requested_cache: Dict[tuple, float] = PrivateAttr(default_factory=dict)
    _pending_request: Optional[tuple] = PrivateAttr(default=None)
    _cache_lock: Lock = PrivateAttr(default_factory=Lock)
    _request_queue: Queue = PrivateAttr(default_factory=Queue)
    _response_queue: Queue = PrivateAttr(default_factory=Queue)
    _stop_event: Event = PrivateAttr(default_factory=Event)
    _session_thread: Optional[Thread] = PrivateAttr(default=None)
    _session_exception: Optional[Exception] = PrivateAttr(default=None)
    _session_finished: bool = PrivateAttr(default=False)
    _result: Optional[OptimizeResult] = PrivateAttr(default=None)
    _completion_warned: bool = PrivateAttr(default=False)
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
        memo[id(self)] = copied
        if self.data is not None:
            copied.data = self.data.copy(deep=True)
        copied._last_candidate = copy.deepcopy(self._last_candidate, memo)
        copied._data_set = self._data_set
        copied._last_outcome = self._last_outcome
        copied._requested_cache = copy.deepcopy(self._requested_cache, memo)
        copied._pending_request = self._pending_request
        copied._cache = copied._build_cache()
        copied._result = copy.deepcopy(self._result, memo)
        copied._completion_warned = self._completion_warned
        return copied

    def __getstate__(self):
        """Snapshot pickle state without stopping the live runtime session."""
        state = super().__getstate__().copy()
        private = dict(state.get("__pydantic_private__", {}) or {})

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
    def result(self) -> Optional[OptimizeResult]:
        """The scipy result after termination, or None while optimization is pending."""
        return self._result

    @property
    def is_done(self) -> bool:
        """Whether scipy has terminated, successfully or otherwise."""
        return self._result is not None

    @property
    def x0(self) -> np.ndarray:
        """Return the optimization start point from config or from the latest dataset row."""
        if self.initial_point is not None:
            missing = [
                k for k in self.vocs.variable_names if k not in self.initial_point
            ]
            if missing:
                raise ValueError(
                    f"initial_point is missing values for variables: {missing}"
                )
            return np.array(
                [self.initial_point[k] for k in self.vocs.variable_names], dtype=float
            )
        return self._get_initial_point()[0]

    def _reset(self):
        """Reset active session state while keeping existing evaluated observations."""
        self._stop_session()
        self._last_outcome = None

    def _set_data(self, data: pd.DataFrame):
        """Replace the full dataset and refresh cache/session-dependent internal state."""
        self._stop_session()
        self.data = data
        self._requested_cache = {}
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
            observed_key = self._point_key(x_value)
            self._cache[observed_key] = self._requested_cache.get(
                observed_key, self._last_outcome
            )
            if self._pending_request is not None:
                self._requested_cache[self._pending_request] = self._last_outcome
                self._cache[self._pending_request] = self._last_outcome
                self._pending_request = None
                if self._session_thread is not None and self._session_thread.is_alive():
                    self._response_queue.put(self._last_outcome)

    def _point_key(self, x: np.ndarray, decimals: int = 12) -> tuple:
        """Convert a floating-point vector to a rounded hashable cache key."""
        return tuple(np.round(np.array(x, dtype=float), decimals=decimals))

    def _build_cache(self) -> dict[tuple, float]:
        """Combine observed history with exact requested-coordinate objective values."""
        if self.data is None or len(self.data) == 0:
            return {}

        # Build a deterministic replay table from prior Xopt observations.
        x_data = get_variable_data(self.vocs, self.data).to_numpy(dtype=float)
        y_data = get_objective_data(self.vocs, self.data).to_numpy()[:, 0]

        cache = {self._point_key(x): float(y) for x, y in zip(x_data, y_data)}
        cache.update(self._requested_cache)
        return cache

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

    def _objective(
        self,
        x: np.ndarray,
        *,
        stop_event: Event,
        request_queue: Queue,
        response_queue: Queue,
        cache: Dict[tuple, float],
        cache_lock: Lock,
        stop_token: object,
    ) -> float:
        """Objective callback used by scipy.

        This method first serves values from cache. For uncached points, it sends
        the requested point to the main thread and blocks until the corresponding
        externally evaluated objective value is provided.
        """
        if stop_event.is_set():
            raise _StopSession()

        key = self._point_key(x)
        with cache_lock:
            if key in cache:
                return cache[key]

        request_queue.put(np.array(x, dtype=float))
        response = response_queue.get()

        if response is stop_token or stop_event.is_set():
            raise _StopSession()

        y_value = float(response)
        with cache_lock:
            cache[key] = y_value
        return y_value

    def _run_session(
        self,
        stop_event: Event,
        request_queue: Queue,
        response_queue: Queue,
        cache: Dict[tuple, float],
        cache_lock: Lock,
        stop_token: object,
    ):
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

        objective = partial(
            self._objective,
            stop_event=stop_event,
            request_queue=request_queue,
            response_queue=response_queue,
            cache=cache,
            cache_lock=cache_lock,
            stop_token=stop_token,
        )
        try:
            result = minimize(objective, self.x0, **minimize_kwargs)
            with cache_lock:
                if self._stop_event is stop_event and not stop_event.is_set():
                    self._result = result
        except _StopSession:
            pass
        except Exception as ex:
            with cache_lock:
                if self._stop_event is stop_event and not stop_event.is_set():
                    self._session_exception = ex
        finally:
            with cache_lock:
                if self._stop_event is stop_event and not stop_event.is_set():
                    self._session_finished = True

    def _start_session_if_needed(self):
        """Start the worker minimize session if one is not currently active."""
        if self.is_done:
            return
        if self._session_thread is not None and self._session_thread.is_alive():
            return

        self._stop_event = Event()
        self._session_exception = None
        self._session_finished = False
        self._request_queue = Queue()
        self._response_queue = Queue()
        self._cache = self._build_cache()

        self._session_thread = Thread(
            target=self._run_session,
            args=(
                self._stop_event,
                self._request_queue,
                self._response_queue,
                self._cache,
                self._cache_lock,
                self._stop_token,
            ),
            daemon=True,
        )
        self._session_thread.start()

    def _stop_session(self):
        """Request worker shutdown and reset transient synchronization primitives."""
        with self._cache_lock:
            self._stop_event.set()
        if self._session_thread is not None and self._session_thread.is_alive():
            self._response_queue.put(self._stop_token)
            self._session_thread.join(timeout=2.0)
            if self._session_thread.is_alive():
                warnings.warn(
                    "scipy worker did not stop within 2 seconds; "
                    "it remains isolated from subsequent sessions.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self._session_thread = None
        self._session_exception = None
        self._session_finished = False
        self._result = None
        self._completion_warned = False
        self._pending_request = None
        self._request_queue = Queue()
        self._response_queue = Queue()
        self._stop_event = Event()

    def _generate(self, first_gen: bool = False) -> Optional[List[Dict[str, float]]]:
        """Return the next candidate requested by the live scipy session.

        The ``first_gen`` argument is accepted to satisfy the sequential generator
        interface; candidate selection always follows the active persistent session.
        """
        self._start_session_if_needed()

        while not self.is_done:
            self._raise_session_error_if_present()

            try:
                requested_x = self._request_queue.get(timeout=0.05)
                self._pending_request = self._point_key(requested_x)
                inputs = dict(zip(self.vocs.variable_names, requested_x.tolist()))
                if self.vocs.constants is not None:
                    inputs.update(self.vocs.constants)
                return [inputs]
            except Empty:
                if self._session_finished:
                    self._raise_session_error_if_present()
                    break

        # Preserve the historical fallback without replaying a completed optimization.
        if self.data is None or len(self.data) == 0:
            raise RuntimeError("scipy minimize converged without available data")

        result = self.result
        if result is None:
            raise RuntimeError("scipy minimize session stopped without a result")

        if not self._completion_warned:
            warnings.warn(
                f"scipy minimize has terminated: {result.message}. "
                "Returning the latest data point; further Xopt.step() calls will "
                "re-evaluate it. Check is_done and result, or call reset() to restart.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._completion_warned = True

        inputs = self.data[self.vocs.variable_names].iloc[-1].to_dict()
        if self.vocs.constants is not None:
            inputs.update(self.vocs.constants)
        return [inputs]
