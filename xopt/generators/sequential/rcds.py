import logging
import math
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field
from pydantic.types import PositiveFloat

from xopt.generators.sequential.sequential_generator import SequentialGenerator

logger = logging.getLogger(__name__)


class RCDS:
    """
    Robust Conjugate Direction Search (RCDS) algorithm.

    Parameters
    ----------
    x0 : array-like
        Initial solution vector.
    init_mat : array-like, optional
        Initial direction matrix. Defaults to None.
    noise : float, optional
        Estimated noise level. Defaults to 0.1.
    step : float, optional
        Step size for the optimization. Defaults to 1e-2.
    tol : float, optional
        Tolerance for convergence. Defaults to 1e-5.

    Attributes
    ----------
    x0 : numpy.matrix
        Initial solution as a column vector.
    Imat : array-like
        Initial direction matrix.
    noise : float
        Estimated noise level.
    step : float
        Step size for the optimization.
    tol : float
        Tolerance for convergence.
    cnt : int
        Internal counter initialized to 0.
    OBJ : None
        Placeholder for the objective function, initialized to None.
    """

    def __init__(
        self,
        x0: np.ndarray,
        init_mat: Optional[np.ndarray] = None,
        noise: float = 0.1,
        step: float = 1e-2,
        tol: float = 1e-5,
    ):
        """
        Initialize the RCDS algorithm.

        Parameters
        ----------
        x0 : np.ndarray
            Initial solution vector.
        init_mat : np.ndarray, optional
            Initial direction matrix. Defaults to None.
        noise : float, optional
            Estimated noise level. Defaults to 0.1.
        step : float, optional
            Step size for the optimization. Defaults to 1e-2.
        tol : float, optional
            Tolerance for convergence. Defaults to 1e-5.
        """
        self.x0 = x0.reshape(-1, 1)  # convert to a col vector
        self.Imat = init_mat
        self.noise = noise
        self.step = step
        self.tol = tol

        # Internal vars
        self.cnt = 0
        self.OBJ = None

    def powellmain(self):
        """
        RCDS main, implementing Powell's direction set update method.

        Returns
        -------
        x1 : numpy.matrix
            The updated solution vector.
        f1 : float
            The function value at the updated solution.
        nf : int
            Number of function evaluations.
        """

        x0 = self.x0
        step = self.step
        tol = self.tol
        self.Nvar = len(x0)
        yield x0
        f0, _ = self.func_obj(x0)
        nf = 1

        xm = x0
        fm = f0

        it = 0
        if self.Imat:
            Dmat = self.Imat
        else:
            Dmat = np.matrix(np.identity(self.Nvar))
        while True:
            logger.debug(f"iteration {it}")
            it += 1
            # step /=1.2

            k = 1
            dl = 0
            for ii in range(self.Nvar):
                dv = Dmat[:, ii]
                gen_gmadp = self.get_min_along_dir_parab(
                    xm, fm, dv, step=step, it=it, idx=ii
                )
                while True:
                    try:
                        yield next(gen_gmadp)
                    except StopIteration as e:
                        x1, f1, ndf = e.value
                        break
                logger.debug(f"best x: {x1}")
                nf += ndf

                if (fm - f1) > dl:
                    dl = fm - f1
                    k = ii
                    logger.debug(
                        "iteration %d, var %d: del = %f updated\n" % (it, ii, dl)
                    )
                fm = f1
                xm = x1

            xt = 2 * xm - x0
            logger.debug("evaluating self.func_obj")
            yield xt
            ft, _ = self.func_obj(xt)
            logger.debug("done")
            nf += 1

            if (
                f0 <= ft
                or 2 * (f0 - 2 * fm + ft) * ((f0 - fm - dl) / (ft - f0)) ** 2 >= dl
            ):
                logger.debug(
                    "   , dir %d not replaced: %d, %d\n"
                    % (
                        k,
                        f0 <= ft,
                        2 * (f0 - 2 * fm + ft) * ((f0 - fm - dl) / (ft - f0)) ** 2
                        >= dl,
                    )
                )
            else:
                ndv = (xm - x0) / np.linalg.norm(xm - x0)
                dotp = np.zeros([self.Nvar])
                logger.debug(dotp)
                for jj in range(self.Nvar):
                    dotp[jj] = abs(np.dot(ndv.transpose(), Dmat[:, jj]))

                if max(dotp) < 0.9:
                    for jj in range(k, self.Nvar - 1):
                        Dmat[:, jj] = Dmat[:, jj + 1]
                    Dmat[:, -1] = ndv

                    # move to the minimum of the new direction
                    dv = Dmat[:, -1]
                    gen_gmadp = self.get_min_along_dir_parab(
                        xm, fm, dv, step=step, it=it, idx=ii
                    )
                    while True:
                        try:
                            yield next(gen_gmadp)
                        except StopIteration as e:
                            x1, f1, ndf = e.value
                            break
                    logger.debug(f"best x: {x1}")
                    nf += ndf
                    logger.debug("end\t%d : %f\n" % (self.cnt, f1))
                    nf = nf + ndf
                    fm = f1
                    xm = x1
                else:
                    logger.debug(
                        "    , skipped new direction %d, max dot product %f\n"
                        % (k, max(dotp))
                    )

            logger.debug("g count is ", self.cnt)

            if 2.0 * abs(f0 - fm) < tol * (abs(f0) + abs(fm)) and tol > 0:
                logger.debug(
                    "terminated: f0=%4.2e\t, fm=%4.2e, f0-fm=%4.2e\n"
                    % (f0, fm, f0 - fm)
                )
                # break

            f0 = fm
            x0 = xm

        return xm, fm, nf

    def get_min_along_dir_parab(
        self,
        x0: np.ndarray,
        f0: float,
        dv: np.ndarray,
        Npmin: int = 6,
        step: Optional[float] = None,
        it: Optional[int] = None,
        idx: Optional[int] = None,
        replaced: bool = False,
    ) -> tuple[np.ndarray, float, int]:
        """
        Find the minimum along a direction using a parabolic fit.

        Parameters
        ----------
        x0 : np.ndarray
            Initial solution vector.
        f0 : float
            Function value at the initial solution.
        dv : np.ndarray
            Direction vector.
        Npmin : int, optional
            Minimum number of points for the line scan. Defaults to 6.
        step : float, optional
            Step size for the bracket minimum. Defaults to None.
        it : int, optional
            Iteration number. Defaults to None.
        idx : int, optional
            Index of the direction. Defaults to None.
        replaced : bool, optional
            Flag indicating if the direction was replaced. Defaults to False.

        Returns
        -------
        x1 : np.ndarray
            The updated solution vector.
        f1 : float
            The function value at the updated solution.
        ndf : int
            Number of function evaluations.
        """
        gen_bm = self.bracketmin(x0, f0, dv, step)
        while True:
            try:
                yield next(gen_bm)
            except StopIteration as e:
                x1, f1, a1, a2, xflist, ndf1 = e.value
                break

        if not replaced:
            logger.debug("iter %d, dir %d: begin\t%d\t%f" % (it, idx, self.cnt, f1))
        else:
            logger.debug(
                "iter %d, new dir %d: begin\t%d\t%f " % (it, idx, self.cnt, f1)
            )

        gen_ls = self.linescan(x1, f1, dv, a1, a2, Npmin, xflist)
        while True:
            try:
                yield next(gen_ls)
            except StopIteration as e:
                x1, f1, ndf2 = e.value
                break

        return x1, f1, ndf1 + ndf2

    def bracketmin(
        self, x0: np.ndarray, f0: float, dv: np.ndarray, step: float
    ) -> tuple[np.ndarray, float, float, float, np.ndarray, int]:
        """
        Bracket the minimum along a direction.

        Parameters
        ----------
        x0 : np.ndarray
            Initial solution vector.
        f0 : float
            Function value at the initial solution.
        dv : np.ndarray
            Direction vector.
        step : float
            Initial step size for the bracket minimum.

        Returns
        -------
        xm : np.ndarray
            The updated solution vector.
        fm : float
            The function value at the updated solution.
        a1 : float
            Lower bound of the bracket.
        a2 : float
            Upper bound of the bracket.
        xflist : np.ndarray
            Array of evaluated points and their function values.
        nf : int
            Number of function evaluations.
        """
        nf = 0
        if math.isnan(f0):
            yield x0
            f0, _ = self.func_obj(x0)
            nf += 1

        xflist = np.array([[0, f0]])
        fm = f0
        am = 0
        xm = x0

        step_init = step

        x1 = x0 + dv * step
        yield x1
        f1, _ = self.func_obj(x1)
        nf += 1

        xflist = np.concatenate((xflist, np.array([[step, f1]])), axis=0)
        if f1 < fm:
            fm = f1
            am = step
            xm = x1

        gold_r = 1.618
        while f1 < fm + self.noise * 3:
            step0 = step
            if abs(step) < 0.1:  # maximum step
                step = step * (1.0 + gold_r)
            else:
                step = step + 0.1
            x1 = x0 + dv * step
            yield x1
            f1, _ = self.func_obj(x1)
            nf += 1

            if math.isnan(f1):
                step = step0
                break
            else:
                xflist = np.concatenate((xflist, np.array([[step, f1]])), axis=0)
                if f1 < fm:
                    fm = f1
                    am = step
                    xm = x1

        a2 = step
        if f0 > fm + self.noise * 3:  # no need to go in the negative direction
            a1 = 0
            a1 = a1 - am
            a2 = a2 - am
            xflist[:, 0] -= am
            return xm, fm, a1, a2, xflist, nf

        # go in the negative direction
        step = -step_init
        x2 = x0 + dv * step
        yield x2
        f2, _ = self.func_obj(x2)
        nf += 1
        xflist = np.concatenate((xflist, np.array([[step, f2]])), axis=0)
        if f2 < fm:
            fm = f2
            am = step
            xm = x2

        while f2 < fm + self.noise * 3:
            step0 = step
            if abs(step) < 0.1:
                step = step * (1.0 + gold_r)
            else:
                step -= 0.1

            x2 = x0 + dv * step
            yield x2
            f2, _ = self.func_obj(x2)
            nf += 1
            if math.isnan(f2):
                step = step0
                break
            else:
                xflist = np.concatenate((xflist, np.array([[step, f2]])), axis=0)
            if f2 < fm:
                fm = f2
                am = step
                xm = x2

        a1 = step
        if a1 > a2:
            a1, a2 = a2, a1

        a1 -= am
        a2 -= am
        xflist[:, 0] -= am
        # sort by alpha
        xflist = xflist[np.argsort(xflist[:, 0])]

        return xm, fm, a1, a2, xflist, nf

    def linescan(
        self,
        x0: np.ndarray,
        f0: float,
        dv: np.ndarray,
        alo: float,
        ahi: float,
        Np: int,
        xflist: np.ndarray,
    ) -> tuple[np.ndarray, float, int]:
        """
        Line optimizer for RCDS.

        Parameters
        ----------
        x0 : np.ndarray
            Initial solution vector.
        f0 : float
            Function value at the initial solution.
        dv : np.ndarray
            Direction vector.
        alo : float
            Lower bound of the bracket.
        ahi : float
            Upper bound of the bracket.
        Np : int
            Number of points for the line scan.
        xflist : np.ndarray
            Array of evaluated points and their function values.

        Returns
        -------
        x1 : np.ndarray
            The updated solution vector.
        f1 : float
            The function value at the updated solution.
        nf : int
            Number of function evaluations.
        """
        nf = 0
        if math.isnan(f0):
            yield x0
            f0, _ = self.func_obj(x0)
            nf += 1

        if alo >= ahi:
            logger.debug(
                "Error: bracket upper bound equal to or lower than lower bound"
            )
            return x0, f0, nf

        if len(x0) != len(dv):
            logger.debug("Error: x0 and dv dimension do not match.")
            return x0, f0, nf

        if math.isnan(Np) or Np < 6:
            Np = 6
        delta = (ahi - alo) / (Np - 1.0)

        alist = np.linspace(alo, ahi, Np)
        flist = alist * float("nan")
        Nlist = np.shape(xflist)[0]
        for ii in range(Nlist):
            if xflist[ii, 1] >= alo and xflist[ii, 1] <= ahi:
                ik = round((xflist[ii, 1] - alo) / delta)
                alist[ik] = xflist[ii, 0]
                flist[ik] = xflist[ii, 1]

        mask = np.ones(len(alist), dtype=bool)
        for ii in range(len(alist)):
            if math.isnan(flist[ii]):
                alpha = alist[ii]
                _x = x0 + alpha * dv
                yield _x
                flist[ii], _ = self.func_obj(_x)
                nf += 1
            if math.isnan(flist[ii]):
                mask[ii] = False

        alist = alist[mask]
        flist = flist[mask]
        if len(alist) <= 0:
            return x0, f0, nf
        elif len(alist) < 5:
            imin = flist.argmin()
            xm = x0 + alist[imin] * dv
            fm = flist[imin]
            return xm, fm, nf
        else:
            p = np.polyfit(alist, flist, 2)
            pf = np.poly1d(p)

            MP = 101
            av = np.linspace(alist[0], alist[-1], MP - 1)
            yv = pf(av)
            imin = yv.argmin()
            xm = x0 + av[imin] * dv
            fm = yv[imin]
            return xm, fm, nf

    def update_obj(self, obj):
        self.OBJ = obj

    def func_obj(self, x, count=True):
        """Objective self.func_objtion for test
        Input:
                x : a column vector
        Output:
                obj : an floating number
        """
        self.Nvar = len(x)
        obj = self.OBJ
        obj_raw = self.OBJ
        if count:
            self.cnt += 1

        return obj, obj_raw


class RCDSGenerator(SequentialGenerator):
    """
    RCDS algorithm.

    Reference:
    An algorithm for online optimization of accelerators
    Huang, X., Corbett, J., Safranek, J. and Wu, J.
    doi: 10.1016/j.nima.2013.05.046

    This algorithm must be stepped serially.

    Attributes
    ----------
    name : str
        Name of the generator.
    x0 : Optional[list]
        Initial solution vector.
    init_mat : Optional[np.ndarray]
        Initial direction matrix.
    noise : PositiveFloat
        Estimated noise level.
    step : PositiveFloat
        Step size for the optimization.
    tol : PositiveFloat
        Tolerance for convergence.
    _ub : np.ndarray
        Upper bounds of the variables.
    _lb : np.ndarray
        Lower bounds of the variables.
    _rcds : RCDS
        Instance of the RCDS algorithm.
    _generator : generator
        Generator for the RCDS algorithm.
    model_config : ConfigDict
        Configuration dictionary for the model.
    """

    name = "rcds"
    init_mat: Optional[np.ndarray] = Field(None)
    noise: PositiveFloat = Field(1e-5)
    step: PositiveFloat = Field(1e-2)
    tol: PositiveFloat = Field(1e-5)

    _rcds: RCDS = None
    _generator = None

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def _reset(self):
        """reset the rcds object"""

        x0, f0 = self._get_initial_point()

        self._rcds = RCDS(
            x0=x0,
            init_mat=self.init_mat,
            noise=self.noise,
            step=self.step,
            tol=self.tol,
        )
        self._rcds.update_obj(float(f0))
        self._generator = self._rcds.powellmain()

    def _add_data(self, new_data: pd.DataFrame):
        # first update the rcds object from the last measurement
        res = float(new_data.iloc[-1][self.vocs.objective_names].to_numpy())

        if self._rcds is not None:
            self._rcds.update_obj(res)

    def _generate(self, first_gen: bool = False):
        """generate a new candidate"""
        x_next = next(self._generator)

        bound_low, bound_up = self.vocs.bounds
        _ub = bound_up
        _lb = bound_low

        # Verify the candidate here
        while np.any(x_next > _ub) or np.any(x_next < _lb):
            self._rcds.update_obj(
                np.nan
            )  # notify RCDS that the search reached the bound
            x_next = next(self._generator)  # request next candidate

        x_next = [float(ele) for ele in x_next]
        return [dict(zip(self.vocs.variable_names, x_next))]
