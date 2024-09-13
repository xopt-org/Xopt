import logging
import math
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field
from pydantic.types import PositiveFloat

from xopt.generator import Generator

logger = logging.getLogger(__name__)


class RCDS:
    def __init__(
        self,
        x0,
        init_mat=None,
        noise=0.1,
        step=1e-2,
        tol=1e-5,
    ):
        """
        Input:
            x0: initial solution
            init_mat: initial direction matrix
            noise: estimated noise level
            step, tol: floating number, step size and tolerance
        """
        self.x0 = np.matrix(x0).T  # convert to a col vector
        self.Imat = init_mat
        self.noise = noise
        self.step = step
        self.tol = tol

        # Internal vars
        self.cnt = 0
        self.OBJ = None

    def powellmain(self):
        """RCDS main, implementing Powell's direction set update method
        Created by X. Huang, 10/5/2016
        Modified by Z. Zhang, 11/30/2022

        Output:
            x1, f1,
            nf: integer, number of evaluations
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
        self, x0, f0, dv, Npmin=6, step=None, it=None, idx=None, replaced=False
    ):
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

    def bracketmin(self, x0, f0, dv, step):
        """bracket the minimum
        Created by X. Huang, 10/5/2016
        Input:
                 self.func_obj is self.func_objtion handle,
                 f0,step : floating number
                 x0, dv: NumPy vector
        Output:
                 xm, fm
                        a1, a2: floating
                        xflist: Nx2 array
                        nf: integer, number of evaluations
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
        #         gold_r = 1.3
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
        # logger.debug(xflist)
        xflist = xflist[np.argsort(xflist[:, 0])]

        return xm, fm, a1, a2, xflist, nf

    def linescan(self, x0, f0, dv, alo, ahi, Np, xflist):
        """Line optimizer for RCDS
        Created by X. Huang, 10/3/2016
        Input:
                 self.func_obj is self.func_objtion handle,
                 f0, alo, ahi: floating number
                 x0, dv: NumPy vector
                 xflist: Nx2 array
        Output:
                 x1, f1, nf
        """
        # global g_noise
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

        if math.isnan(Np) | (Np < 6):
            Np = 6
        delta = (ahi - alo) / (Np - 1.0)

        #         alist = np.arange(alo,ahi,(ahi-alo)/(Np-1))
        alist = np.linspace(alo, ahi, Np)
        flist = alist * float("nan")
        Nlist = np.shape(xflist)[0]
        for ii in range(Nlist):
            if xflist[ii, 1] >= alo and xflist[ii, 1] <= ahi:
                ik = round((xflist[ii, 1] - alo) / delta)
                # logger.debug('test', ik, ii, len(alist),len(xflist),xflist[ii,0])
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

        # filter out NaNs
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
            # logger.debug(np.c_[alist,flist])
            (p) = np.polyfit(alist, flist, 2)
            pf = np.poly1d(p)

            # remove outlier and re-fit here, to be done later

            MP = 101
            av = np.linspace(alist[0], alist[-1], MP - 1)
            yv = pf(av)
            imin = yv.argmin()
            xm = x0 + av[imin] * dv
            fm = yv[imin]
            # logger.debug(x0, xm, fm)
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


class RCDSGenerator(Generator):
    """
    RCDS algorithm.

    Reference:
    An algorithm for online optimization of accelerators
    Huang, X., Corbett, J., Safranek, J. and Wu, J.
    doi: 10.1016/j.nima.2013.05.046

    This algorithm must be stepped serially.
    """

    name = "rcds"
    x0: Optional[list] = Field(None)
    init_mat: Optional[np.ndarray] = Field(None)
    noise: PositiveFloat = Field(1e-5)
    step: PositiveFloat = Field(1e-2)
    tol: PositiveFloat = Field(1e-5)

    _ub = 0
    _lb = 0
    _rcds: RCDS = None
    _generator = None

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        bound_low, bound_up = self.vocs.bounds
        self._ub = bound_up
        self._lb = bound_low
        x_ave = (bound_up + bound_low) / 2
        if self.x0 is None:
            x0 = x_ave
        else:
            x0 = self.x0

        self._rcds = RCDS(
            x0=x0,
            init_mat=self.init_mat,
            noise=self.noise,
            step=self.step,
            tol=self.tol,
        )
        self._generator = self._rcds.powellmain()

    def add_data(self, new_data: pd.DataFrame):
        assert (
            len(new_data) == 1
        ), f"length of new_data must be 1, found: {len(new_data)}"
        res = self.vocs.objective_data(new_data).to_numpy()
        assert res.shape == (1, 1)
        obj = res[0, 0]
        self._rcds.update_obj(obj)

    def generate(self, n_candidates) -> list[dict]:
        if n_candidates != 1:
            raise NotImplementedError("rcds can only produce one candidate at a time")

        x_next = np.array(next(self._generator))  # note that x_next is a np.matrix!

        # Verify the candidate here
        while np.any(x_next > self._ub) or np.any(x_next < self._lb):
            self._rcds.update_obj(
                np.nan
            )  # notify RCDS that the search reached the bound
            x_next = np.array(next(self._generator))  # request next candidate

        # Return the next value
        try:
            pd.DataFrame(dict(zip(self.vocs.variable_names, x_next)))
        except Exception as e:
            print(self.vocs.variable_names, x_next, type(x_next), x_next.shape)
            raise e

        return [dict(zip(self.vocs.variable_names, x_next))]
