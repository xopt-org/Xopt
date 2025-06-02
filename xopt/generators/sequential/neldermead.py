import logging
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, field_validator

from xopt.generators.sequential.sequential_generator import SequentialGenerator
from xopt.pydantic import XoptBaseModel
from xopt.vocs import VOCS, validate_variable_bounds

logger = logging.getLogger(__name__)


class SimplexState(XoptBaseModel):
    """Container model for all simplex parameters"""

    astg: int = -1
    N: Optional[int] = None
    kend: int = 0
    jend: int = 0
    ind: Optional[np.ndarray] = None
    sim: Optional[np.ndarray] = None
    fsim: Optional[np.ndarray] = None
    fxr: Optional[float] = None
    x: Optional[np.ndarray] = None
    xr: Optional[np.ndarray] = None
    xe: Optional[np.ndarray] = None
    xc: Optional[np.ndarray] = None
    xcc: Optional[np.ndarray] = None
    xbar: Optional[np.ndarray] = None
    doshrink: int = 0
    ngen: int = 0
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator(
        "ind", "fsim", "sim", "x", "xr", "xe", "xc", "xcc", "xbar", mode="before"
    )
    def to_numpy(cls, v):
        return np.array(v, dtype=np.float64)


STATE_KEYS = [
    "astg",
    "N",
    "kend",
    "jend",
    "ind",
    "sim",
    "fsim",
    "fxr",
    "x",
    "xr",
    "xe",
    "xc",
    "xcc",
    "xbar",
    "doshrink",
    "ngen",
]


class NelderMeadGenerator(SequentialGenerator):
    """
    Nelder-Mead algorithm from SciPy in Xopt's Generator form.
    Converted to use a state machine to resume in exactly the last state.

    Attributes
    ----------
    name : str
        The name of the generator.
    initial_point : Optional[Dict[str, float]]
        Initial point for the optimization.
    initial_simplex : Optional[Dict[str, Union[List[float], np.ndarray]]]
        Initial simplex for the optimization.
    adaptive : bool
        Change hyperparameters based on dimensionality.
    current_state : SimplexState
        Current state of the simplex.
    future_state : Optional[SimplexState]
        Future state of the simplex.
    x : Optional[np.ndarray]
        Current x value.
    y : Optional[float]
        Current y value.

    Methods
    -------
    add_data(self, new_data: pd.DataFrame)
        Add new data to the generator.
    generate(self, n_candidates: int) -> Optional[List[Dict[str, float]]]
        Generate a specified number of candidate samples.
    _call_algorithm(self)
        Call the Nelder-Mead algorithm.
    simplex(self) -> Dict[str, np.ndarray]
        Returns the simplex in the current state.
    """

    name = "neldermead"
    supports_single_objective: bool = True

    initial_point: Optional[Dict[str, float]] = None  # replaces x0 argument
    initial_simplex: Optional[Dict[str, Union[List[float], np.ndarray]]] = (
        None  # This overrides the use of initial_point
    )
    # Same as scipy.optimize._optimize._minimize_neldermead
    adaptive: bool = Field(
        True, description="Change hyperparameters based on dimensionality"
    )
    current_state: SimplexState = SimplexState()
    future_state: Optional[SimplexState] = None

    # Internal data structures
    x: Optional[np.ndarray] = None
    y: Optional[float] = None
    manual_data_cnt: int = Field(
        0, description="How many points are considered manual/not part of simplex run"
    )

    _initial_simplex = None
    _initial_point = None
    _saved_options: Dict = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._saved_options = self.model_dump(
            exclude={"current_state", "future_state"}
        ).copy()  # Used to keep track of changed options

        if self.initial_simplex:
            self._initial_simplex = np.array(
                [self.initial_simplex[k] for k in self.vocs.variable_names]
            ).T
        else:
            self._initial_simplex = None

    def _generate(self, first_gen: bool = False) -> Optional[List[Dict[str, float]]]:
        """
        Generate candidate.

        Returns
        -------
        Optional[List[Dict[str, float]]]
            A list of dictionaries containing the generated samples.
        """

        if self.current_state.N is None:
            # fresh start
            pass
        else:
            n_inputs = len(self.data)
            if self.current_state.ngen == n_inputs:
                # We are in a state where result of last point is known
                pass
            else:
                pass

        results = self._call_algorithm()

        x, state_extra = results
        assert len(state_extra) == len(STATE_KEYS)
        stateobj = SimplexState(**{k: v for k, v in zip(STATE_KEYS, state_extra)})
        self.future_state = stateobj

        inputs = dict(zip(self.vocs.variable_names, x))
        if self.vocs.constants is not None:
            inputs.update(self.vocs.constants)

        return [inputs]

    @property
    def x0(self) -> np.ndarray:
        """
        Raw internal initial point for convenience.
        If initial_point is set, it will be used. Otherwise, if data is set, the last point will be used.

        Returns
        -------
        np.ndarray
            The initial point as a NumPy array.
        """
        if self._initial_point is not None:
            return self._initial_point
        elif self.initial_point is not None:
            return np.array([self.initial_point[k] for k in self.vocs.variable_names])
        elif self.data is not None:
            return self._get_initial_point()[0]
        else:
            raise ValueError(
                "No initial point specified in generator or taken from data"
            )

    def _add_data(self, new_data: pd.DataFrame):
        """
        Add new data to the generator.

        Parameters
        ----------
        new_data : pd.DataFrame
            The new data to be added.
        """
        if len(new_data) == 0:
            # empty data, i.e. no steps yet
            assert self.future_state is None
            return

        ndata = len(self.data)
        ngen = self.current_state.ngen

        # Complicated part - need to determine if data corresponds to result of last gen
        if not self.is_active:
            assert self.future_state is None, "Not active, but future state exists?"

            variable_data = self.vocs.variable_data(self.data).to_numpy()
            objective_data = self.vocs.objective_data(self.data).to_numpy()[:, 0]

            _initial_simplex = variable_data.copy()
            N = self.vocs.n_variables
            if _initial_simplex.shape[0] >= N + 1:
                # TODO: is this really needed?
                # if we have enough, form new simplex and force state to just after it is all probed
                _initial_simplex = _initial_simplex[-(N + 1) :, :]
                objective_data = objective_data[-(N + 1) :]

                fake_initialized_state = _fake_partial_state_gen(
                    _initial_simplex, objective_data
                )
                self.current_state = fake_initialized_state
                self._initial_simplex = _initial_simplex
                self.manual_data_cnt = len(self.data) - (N + 1)
            else:
                # otherwise, just set the last point as the initial point - effectively loses output data
                self.current_state = SimplexState()
                self._initial_simplex = None
                self.manual_data_cnt = len(self.data)

            self._initial_point = _initial_simplex[-1, :]
            self.y = float(objective_data[-1])
        else:
            assert new_data.shape[0] == 1, (
                "Only one point at a time is expected in active mode"
            )
            # new data -> advance state machine 1 step
            n_auto_points = ndata - self.manual_data_cnt
            assert n_auto_points == self.future_state.ngen, (
                f"Internal error {n_auto_points=} {self.future_state=} {ndata=}"
            )
            assert n_auto_points == ngen + 1, f"Internal error {n_auto_points=} {ngen=}"
            self.current_state = self.future_state
            self.future_state = None

            # Can have multiple points if resuming from file, grab last one
            new_data_df = self.vocs.objective_data(new_data)
            res = new_data_df.iloc[-1:, :].to_numpy()
            assert np.shape(res) == (1, 1), f"Bad last point [{res}]"

            yt = res[0, 0].item()

            self.y = yt

    def _set_data(self, data: pd.DataFrame):
        # just store data
        self.data = data

        new_data_df = self.vocs.objective_data(data)
        res = new_data_df.iloc[-1:, :].to_numpy()
        assert np.shape(res) == (1, 1), f"Bad last point [{res}]"

    def _reset(self):
        self.current_state = SimplexState()
        self.future_state = None
        if self.initial_simplex:
            self._initial_simplex = np.array(
                [self.initial_simplex[k] for k in self.vocs.variable_names]
            ).T
        else:
            self._initial_simplex = None

    def _call_algorithm(self):
        mins, maxs = self.vocs.bounds
        results = _neldermead_generator(
            x0=self.x0,
            state=self.current_state,
            lastval=self.y,
            adaptive=self.adaptive,
            initial_simplex=self._initial_simplex,
            bounds=(mins, maxs),
        )

        self.y = None
        return results

    @property
    def simplex(self) -> Dict[str, np.ndarray]:
        """
        Returns the simplex in the current state.

        Returns
        -------
        Dict[str, np.ndarray]
            The simplex in the current state.
        """
        sim = self.current_state.sim
        return dict(zip(self.vocs.variable_names, sim.T))


def _fake_partial_state_gen(sim: np.ndarray, fsim: np.ndarray):
    # active stage
    # internal simplex variables that will be saved/restored
    ind = fxr = xr = xbar = x = xe = xc = xcc = None
    jend = 0
    doshrink = 0

    assert sim.shape[0] == fsim.shape[0]
    kend = sim.shape[0] - 1
    ngen = sim.shape[0]
    astg = -1
    # lastval will set final fsim entry

    state = SimplexState(
        astg=astg,
        N=sim.shape[1],
        kend=kend,
        jend=jend,
        ind=ind,
        sim=sim,
        fsim=fsim,
        fxr=fxr,
        x=x,
        xr=xr,
        xe=xe,
        xc=xc,
        xcc=xcc,
        xbar=xbar,
        doshrink=doshrink,
        ngen=ngen,
    )
    return state


def _neldermead_generator(
    x0: np.ndarray,
    state: SimplexState,
    lastval: Optional[float] = None,
    initial_simplex: Optional[np.ndarray] = None,
    adaptive: bool = True,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Optional[Tuple[np.ndarray, Tuple]]:
    """
    Modification of scipy.optimize._optimize._minimize_neldermead
    https://github.com/scipy/scipy/blob/4cf21e753cf937d1c6c2d2a0e372fbc1dbbeea81/scipy/optimize/_optimize.py#L635

    Modified to work in stages, and to return the state at each stage.

    Original SciPy docstring:

    Minimization of scalar function of one or more variables using the
    Nelder-Mead algorithm.

    Parameters
    ----------
    x0 : np.ndarray
        Initial point for the optimization.
    state : SimplexState
        Current state of the simplex.
    lastval : float, optional
        Last function value.
    initial_simplex : np.ndarray, optional
        Initial simplex for the optimization.
    adaptive : bool, optional
        Adapt algorithm parameters to dimensionality of problem. Useful for high-dimensional minimization.
    bounds : Tuple[np.ndarray, np.ndarray], optional
        Bounds on variables. There are two ways to specify the bounds:
            1. Instance of `Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None is used to specify no bound.

    Returns
    -------
    Optional[Tuple[np.ndarray, Tuple]]
        The next point to evaluate and the state of the simplex.
    """

    # Stages
    # -1 - default (normal) state
    # 0 during initial simplex
    # 1-5 during loop

    # TRACE flag is for performance, since it will prevent python message formatting if False
    TRACE = False

    def log(s):
        print(s)

    # active stage
    astg = 0
    # internal simplex variables that will be saved/restored
    ind = fxr = xr = xbar = x = xe = xc = xcc = None
    kend = jend = ngen = 0
    doshrink = 0

    def save_state():
        # nonlocal ngen
        # ngen += 1
        return (
            astg,
            N,
            kend,
            jend,
            ind,
            sim,
            fsim,
            fxr,
            x,
            xr,
            xe,
            xc,
            xcc,
            xbar,
            doshrink,
            ngen + 1,
        )

    (
        astg,
        N,
        kend,
        jend,
        ind,
        sim,
        fsim,
        fxr,
        x,
        xr,
        xe,
        xc,
        xcc,
        xbar,
        doshrink,
        ngen,
    ) = (getattr(state, k) for k in STATE_KEYS)
    stage = state.astg
    if stage != -1:
        assert lastval is not None
    astg = 0

    if bounds is not None:
        lower_bound, upper_bound = bounds
        # check bounds
        if (lower_bound > upper_bound).any():
            raise ValueError(
                "Nelder Mead - one of the lower bounds is greater than an upper bound."
            )
        if np.any(lower_bound > x0) or np.any(x0 > upper_bound):
            warnings.warn("Initial guess is not within the specified bounds")

    if stage == -1:
        x0 = np.array(x0, dtype=float).flatten()

        nonzdelt = 0.05
        zdelt = 0.00025

        if bounds is not None:
            x0 = np.clip(x0, lower_bound, upper_bound)

        if initial_simplex is None:
            N = len(x0)

            sim = np.empty((N + 1, N), dtype=x0.dtype)
            sim[0] = x0
            for k in range(N):
                y = np.array(x0, copy=True)
                if y[k] != 0:
                    y[k] = (1 + nonzdelt) * y[k]
                else:
                    y[k] = zdelt
                sim[k + 1] = y
        else:
            sim = np.array(initial_simplex, dtype=float).copy()
            if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
                raise ValueError(
                    "`initial_simplex` should be an array of shape (N+1,N)"
                )
            if len(x0) != sim.shape[1]:
                raise ValueError(
                    "Size of `initial_simplex` is not consistent with `x0`"
                )
            N = sim.shape[1]

        if bounds is not None:
            sim = np.clip(sim, lower_bound, upper_bound)

        fsim = np.full((N + 1,), np.nan, dtype=float)

    for k in range(kend, N + 1):
        if stage == -1:
            state = save_state()
            if TRACE:
                log(f"Stage 0 yield {sim[k]=} {stage=} {state=}")
            return sim[k], state
        else:
            if TRACE:
                log(f"Stage 0 resume {sim[k]=} {stage=} {state=} {lastval=}")
            stage = -1
            fsim[k] = lastval
            lastval = None
            kend += 1

    if stage == -1:
        ind = np.argsort(fsim)
        sim = np.take(sim, ind, 0)
        fsim = np.take(fsim, ind, 0)

        ind = np.argsort(fsim)
        fsim = np.take(fsim, ind, 0)
        # sort so sim[0,:] has the lowest function value
        sim = np.take(sim, ind, 0)

    if adaptive:
        dim = float(len(x0))
        rho = 1
        chi = 1 + 2 / dim
        psi = 0.75 - 1 / (2 * dim)
        sigma = 1 - 1 / dim
    else:
        rho = 1
        chi = 2
        psi = 0.5
        sigma = 0.5

    one2np1 = list(range(1, N + 1))
    assert stage >= 1 or stage == -1

    while True:
        if stage == -1:
            astg = 1

            xbar = np.add.reduce(sim[:-1], 0) / N
            xr = (1 + rho) * xbar - rho * sim[-1]
            if bounds is not None:
                xr = np.clip(xr, lower_bound, upper_bound)

            state = save_state()
            if TRACE:
                log(f"Stage 1 yield {xr=} {stage=} {state=}")
            return xr, state
        elif stage == 1:
            if TRACE:
                log(f"Stage 1 resume {xr=} {stage=} {state=} {lastval=}")
            stage = -1
            fxr = lastval
            lastval = None
        else:
            pass

        if stage == -1:
            doshrink = 0

        if fxr < fsim[0] or stage == 2:
            astg = 2
            if stage == -1:
                xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
                if bounds is not None:
                    xe = np.clip(xe, lower_bound, upper_bound)
                state = save_state()
                if TRACE:
                    log(f"Stage 2 yield {xe=} {stage=} {state=}")
                return xe, state
            elif stage == 2:
                if TRACE:
                    log(f"Stage 2 resume {xe=} {stage=} {state=}")
                stage = -1
                fxe = lastval
                lastval = None
            else:
                raise Exception("Simplex state is wrong")

            if fxe < fxr:
                sim[-1] = xe
                fsim[-1] = fxe
            else:
                sim[-1] = xr
                fsim[-1] = fxr

        else:  # fsim[0] <= fxr
            if fxr < fsim[-2] and stage == -1:
                sim[-1] = xr
                fsim[-1] = fxr
            else:  # fxr >= fsim[-2]
                # Perform contraction
                if (stage == -1 and fxr < fsim[-1]) or stage == 3:
                    astg = 3
                    if stage == -1:
                        xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                        if bounds is not None:
                            xc = np.clip(xc, lower_bound, upper_bound)
                        state = save_state()
                        if TRACE:
                            log(f"Stage 3 yield {xc=} {stage=} {state=}")
                        return xc, state
                    elif stage == 3:
                        if TRACE:
                            log(f"Stage 3 resume {xc=} {stage=} {state=}")
                        stage = -1
                        fxc = lastval
                        lastval = None
                    else:
                        raise Exception("Simplex state is wrong")

                    if fxc <= fxr:
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        doshrink = 1
                elif stage == -1 or stage == 4:
                    astg = 4
                    if stage == -1:
                        # Perform an inside contraction
                        xcc = (1 - psi) * xbar + psi * sim[-1]
                        if bounds is not None:
                            xcc = np.clip(xcc, lower_bound, upper_bound)
                        state = save_state()
                        if TRACE:
                            log(f"Stage 4 yield {xcc=} {stage=} {state=}")
                        return xcc, state
                    elif stage == 4:
                        if TRACE:
                            log(f"Stage 4 resume {xcc=} {stage=} {state=}")
                        stage = -1
                        fxcc = lastval
                        lastval = None
                    else:
                        raise Exception("Simplex state is wrong")

                    if fxcc < fsim[-1]:
                        sim[-1] = xcc
                        fsim[-1] = fxcc
                    else:
                        doshrink = 1
                else:
                    assert stage == 5

                assert stage == -1 or stage == 5
                if (stage == -1 and doshrink) or stage == 5:
                    astg = 5
                    for jidx in range(jend, len(one2np1)):
                        j = one2np1[jidx]
                        if stage == -1:
                            sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                            if bounds is not None:
                                sim[j] = np.clip(sim[j], lower_bound, upper_bound)
                            state = save_state()
                            if TRACE:
                                log(f"Stage 5 yield {sim[j]=} {stage=} {state=}")
                            return sim[j], state
                        else:
                            if TRACE:
                                log(f"Stage 5 resume {sim[j]=} {stage=} {state=}")
                            stage = -1
                            fsim[j] = lastval
                            lastval = None
                        jend += 1
                    jend = 0

        ind = np.argsort(fsim)
        sim = np.take(sim, ind, 0)
        fsim = np.take(fsim, ind, 0)

    def simplex_inputs(
        vocs: VOCS,
        x0: np.ndarray,
        custom_bounds: dict = None,
        nonzdelt: float = 0.05,
        zdelt: float = 0.00025,
    ) -> np.ndarray:
        assert x0.ndim == 1, "Expected 1D array"
        assert len(x0) == vocs.n_variables, (
            f"Expected {vocs.n_variables} dimensions, got {len(x0)}"
        )
        if custom_bounds is None:
            bounds = vocs.variables
        else:
            variable_bounds = pd.DataFrame(vocs.variables)

            if not isinstance(custom_bounds, dict):
                raise TypeError("`custom_bounds` must be a dict")

            try:
                validate_variable_bounds(custom_bounds)
            except ValueError:
                raise ValueError("specified `custom_bounds` not valid")

            old_custom_bounds = deepcopy(custom_bounds)

            custom_bounds = pd.DataFrame(custom_bounds)
            custom_bounds = custom_bounds.clip(
                variable_bounds.iloc[0], variable_bounds.iloc[1], axis=1
            )
            bounds = custom_bounds.to_dict()

            for name, value in bounds.items():
                if value[0] == value[1]:
                    raise ValueError(
                        f"specified `custom_bounds` for {name} is outside vocs domain"
                    )

            if bounds != old_custom_bounds:
                warnings.warn(
                    "custom bounds were clipped by vocs bounds", RuntimeWarning
                )

            for k in bounds.keys():
                bounds[k] = [bounds[k][i] for i in range(2)]

        lower_bound, upper_bound = bounds

        N = len(x0)
        sim = np.empty((N + 1, N), dtype=x0.dtype)
        sim[0] = x0
        for k in range(N):
            y = np.array(x0, copy=True)
            if y[k] != 0:
                y[k] = (1 + nonzdelt) * y[k]
            else:
                y[k] = zdelt
            sim[k + 1] = y

        if bounds is not None:
            sim = np.clip(sim, lower_bound, upper_bound)

        return sim
