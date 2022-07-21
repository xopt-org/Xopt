from abc import abstractmethod

import logging
import warnings

from typing import Dict, List

from numpy import shape, asfarray

import numpy as np

import pandas as pd

from xopt.generator import Generator, GeneratorOptions

logger = logging.getLogger(__name__)


class ScipyOptimizeGenerator(Generator):
    """
    Base class for scipy.optimize routines that have been converted to generator form.

    These algorithms must be stepped serially.
    """

    alias = None

    def __init__(self, vocs, options):
        super().__init__(vocs, options)

        # Internal data structures
        self.y = None  # Used to coordinate with func
        self._lock = False  # mechanism to lock function calls
        self._algorithm = None  # Will initialize on first generate

        self.initial_point = options.initial_point  # Handles None also.

        self._saved_options = options.copy()  # Used to keep track of changed options

    # Wrapper to refer to internal data
    def func(self, x):
        assert np.array_equal(x, self.x), f"{x} should equal {self.x}"
        return self.y

    @property
    def x0(self):
        """Raw internal initial point for convenience"""
        return np.array(
            [self.options.initial_point[k] for k in self.vocs.variable_names]
        )

    @abstractmethod
    def _init_algorithm(self):
        """
        sets self._algorithm to the generator function (initializing it).
        """
        pass
        # Initialize the generator
        # self._algorithm = self._generator_function(
        #    self.func,
        #    self.x0,
        #    **kwargs)
        #

    @property
    def initial_point(self):
        return self.options.initial_point

    @initial_point.setter
    def initial_point(self, value):
        if value is None:
            value = self.vocs.random_inputs(
                include_constants=False, include_linked_variables=False
            )
        self.options.initial_point = value

    def add_data(self, new_data: pd.DataFrame):
        assert (
            len(new_data) == 1
        ), f"length of new_data must be 1, found: {len(new_data)}"
        res = self.vocs.objective_data(new_data).to_numpy()
        assert shape(res) == (1, 1)
        y = res[0, 0]
        if np.isinf(y) or np.isnan(y):
            self._is_done = True
            return

        self.y = y  # generator_function accesses this
        self.data = new_data
        self._lock = False  # unlock

    def generate(self, n_candidates) -> List[Dict]:

        # Check if any options were changed from init. If so, reset the algorithm
        if self.options != self._saved_options:
            self._algorithm = None
            self._saved_options = self.options.copy()

        # Actually start the algorithm.
        if self._algorithm is None:
            self._init_algorithm()
            self._is_done = False

        if self.is_done:
            return None

        if self._lock:
            raise ValueError(
                "Generation is locked via ._lock. "
                "Please call `add_data` before any further generate(1)"
            )

        if n_candidates != 1:
            raise NotImplementedError(
                "simplex can only produce one candidate at a time"
            )

        try:
            self.x, self.state = next(
                self._algorithm
            )  # raw x point and any state to be passed back
            self._lock = True
            self.inputs = [dict(zip(self.vocs.variable_names, self.x))]
        except StopIteration:
            self._is_done = True

        return self.inputs


class NelderMeadOptions(GeneratorOptions):
    initial_point: Dict[str, float] = None  # replaces x0 argument
    initial_simplex: Dict[
        str, List[float]
    ] = None  # This overrides the use of initial_point
    # Same as scipy.optimize._optimize._minimize_neldermead
    adaptive: bool = True
    xatol: float = 1e-4
    fatol: float = 1e-4


class NelderMeadGenerator(ScipyOptimizeGenerator):
    """
    Nelder-Mead algorithm from SciPy in Xopt's Generator form.
    """

    alias = "neldermead"

    @staticmethod
    def default_options() -> NelderMeadOptions:
        return NelderMeadOptions()

    def __init__(self, vocs, options: NelderMeadOptions = NelderMeadOptions()):
        super().__init__(vocs, options)

    def _init_algorithm(self):
        """
        sets self._algorithm to the generator function (initializing it).
        """

        options = self.options  # convenience

        if options.initial_simplex:
            sim = np.array(
                [self.options.initial_simplex[k] for k in self.vocs.variable_names]
            ).T
        else:
            sim = None

        self._algorithm = _neldermead_generator(  # adapted from scipy.optimize
            self.func,  # Handled by base class
            self.x0,  # Handled by base class
            adaptive=self.options.adaptive,
            xatol=self.options.xatol,
            fatol=self.options.fatol,
            initial_simplex=sim,
            bounds=self.vocs.bounds,
        )

    @property
    def simplex(self):
        """
        Returns the simplex in the current state.
        """
        sim = self.state
        return dict(zip(self.vocs.variable_names, sim.T))


def _neldermead_generator(
    func,
    x0,
    # args=(), callback=None,
    # maxiter=None, maxfev=None, disp=False,
    # return_all=False,
    initial_simplex=None,
    xatol=1e-4,
    fatol=1e-4,
    adaptive=True,
    bounds=None,
):
    """
    Modification of scipy.optimize._optimize._minimize_neldermead
    https://github.com/scipy/scipy/blob/4cf21e753cf937d1c6c2d2a0e372fbc1dbbeea81/scipy/optimize/_optimize.py#L635

    `yield x, sim` is inserted before every call to func(x)
    This converts this function into a generator.

    Original SciPy docstring:

    Minimization of scalar function of one or more variables using the
    Nelder-Mead algorithm.
    Options
    -------
    maxiter, maxfev : int
        Maximum allowed number of iterations and function evaluations.
        Will default to ``N*200``, where ``N`` is the number of
        variables, if neither `maxiter` or `maxfev` is set. If both
        `maxiter` and `maxfev` are set, minimization will stop at the
        first reached.
    initial_simplex : array_like of shape (N + 1, N)
        Initial simplex. If given, overrides `x0`.
        ``initial_simplex[j,:]`` should contain the coordinates of
        the jth vertex of the ``N+1`` vertices in the simplex, where
        ``N`` is the dimension.
    xatol : float, optional
        Absolute error in xopt between iterations that is acceptable for
        convergence.
    fatol : number, optional
        Absolute error in func(xopt) between iterations that is acceptable for
        convergence.
    adaptive : bool, optional
        Adapt algorithm parameters to dimensionality of problem. Useful for
        high-dimensional minimization [1]_.
    bounds : sequence or `Bounds`, optional
        Bounds on variables. There are two ways to specify the bounds:
            1. Instance of `Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.
        Note that this just clips all vertices in simplex based on
        the bounds.
    References
    ----------
    .. [1] Gao, F. and Han, L.
       Implementing the Nelder-Mead simplex algorithm with adaptive
       parameters. 2012. Computational Optimization and Applications.
       51:1, pp. 259-277
    """

    x0 = asfarray(x0).flatten()

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

    nonzdelt = 0.05
    zdelt = 0.00025

    if bounds is not None:
        lower_bound, upper_bound = bounds  # was: bounds.lb, bounds.ub
        # check bounds
        if (lower_bound > upper_bound).any():
            raise ValueError(
                "Nelder Mead - one of the lower bounds is greater than an upper bound."
            )
        if np.any(lower_bound > x0) or np.any(x0 > upper_bound):
            warnings.warn("Initial guess is not within the specified bounds")

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
        sim = np.asfarray(initial_simplex).copy()
        if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
            raise ValueError("`initial_simplex` should be an array of shape (N+1,N)")
        if len(x0) != sim.shape[1]:
            raise ValueError("Size of `initial_simplex` is not consistent with `x0`")
        N = sim.shape[1]

    if bounds is not None:
        sim = np.clip(sim, lower_bound, upper_bound)

    one2np1 = list(range(1, N + 1))
    fsim = np.full((N + 1,), np.inf, dtype=float)

    for k in range(N + 1):
        x = sim[k]
        yield x, sim
        fsim[k] = func(x)

    ind = np.argsort(fsim)
    sim = np.take(sim, ind, 0)
    fsim = np.take(fsim, ind, 0)

    ind = np.argsort(fsim)
    fsim = np.take(fsim, ind, 0)
    # sort so sim[0,:] has the lowest function value
    sim = np.take(sim, ind, 0)

    # while (fcalls[0] < maxfun and iterations < maxiter):
    while True:

        if (
            np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xatol
            and np.max(np.abs(fsim[0] - fsim[1:])) <= fatol
        ):
            break

        xbar = np.add.reduce(sim[:-1], 0) / N
        xr = (1 + rho) * xbar - rho * sim[-1]
        if bounds is not None:
            xr = np.clip(xr, lower_bound, upper_bound)
        yield xr, sim
        fxr = func(xr)
        doshrink = False

        if fxr < fsim[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
            if bounds is not None:
                xe = np.clip(xe, lower_bound, upper_bound)
            yield xe, sim
            fxe = func(xe)

            if fxe < fxr:
                sim[-1] = xe
                fsim[-1] = fxe
            else:
                sim[-1] = xr
                fsim[-1] = fxr
        else:  # fsim[0] <= fxr
            if fxr < fsim[-2]:
                sim[-1] = xr
                fsim[-1] = fxr
            else:  # fxr >= fsim[-2]
                # Perform contraction
                if fxr < fsim[-1]:
                    xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                    if bounds is not None:
                        xc = np.clip(xc, lower_bound, upper_bound)
                    yield xc, sim
                    fxc = func(xc)

                    if fxc <= fxr:
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        doshrink = True
                else:
                    # Perform an inside contraction
                    xcc = (1 - psi) * xbar + psi * sim[-1]
                    if bounds is not None:
                        xcc = np.clip(xcc, lower_bound, upper_bound)
                    yield xcc, sim
                    fxcc = func(xcc)

                    if fxcc < fsim[-1]:
                        sim[-1] = xcc
                        fsim[-1] = fxcc
                    else:
                        doshrink = True

                if doshrink:
                    for j in one2np1:
                        sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                        if bounds is not None:
                            sim[j] = np.clip(sim[j], lower_bound, upper_bound)
                        x = sim[j]
                        yield x, sim
                        fsim[j] = func(x)

        ind = np.argsort(fsim)
        sim = np.take(sim, ind, 0)
        fsim = np.take(fsim, ind, 0)
