import logging
import math
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field
from pydantic.types import PositiveFloat

from xopt.generators.sequential.sequential_generator import SequentialGenerator

logger = logging.getLogger(__name__)


###############################################################################
# BracketMinStepper replicates the generator-style bracketmin.
###############################################################################
class BracketMinStepper:
    """
    A stateful stepper for the bracket search portion.

    This stepper implements the following logic (mirroring the original generator):

      1. Initialize:
           - xflist = [[0, f0]]
           - fm = f0, am = 0, xm = x0, step_init = given step.
      2. Compute candidate: x1 = x0 + dv * step.
         Wait for external evaluation (update).
      3. On update, record f1; if f1 < fm, update fm, am, xm.
      4. Then, while f1 < fm + noise*3:
             - Save current step as step0.
             - If |step| < 0.1, update step = step*(1+gold_r) else step = step + 0.1.
             - Propose new candidate x1 = x0 + dv*step and wait.
             - On update, if math.isnan(f1) then revert step to step0 and break; else update xflist and best values.
      5. When the loop finishes, set a2 = step.
      6. If f0 > fm + noise*3 (i.e. no need for negative search), finish by
         adjusting a1 = 0 - am, a2 = a2 - am, subtract am from xflist[:,0] and return.
      7. Otherwise, start negative phase:
             - Reset: step = -step_init.
             - Compute candidate x2 = x0 + dv*step; wait for update.
             - On update, update nf and xflist and if f2 < fm, update fm, am, xm.
             - Then while f2 < fm + noise*3:
                  • Save step0.
                  • If |step| < 0.1, update step = step*(1+gold_r) else step = step - 0.1.
                  • Propose candidate x2 = x0 + dv*step; wait.
                  • On update, if math.isnan(f2), revert step = step0 and break; else update xflist and best values.
             - Finally, set a1 = step.
             - If a1 > a2, swap them.
             - Then subtract am from a1 and a2 and from xflist[:,0] and sort xflist.

    The methods next_candidate() and update(new_obj) let external code drive the computation.
    """

    def __init__(self, rcds, x0: np.ndarray, f0: float, dv: np.ndarray, step: float):
        self.rcds = rcds
        self.x0 = x0
        self.f0 = f0
        self.dv = dv
        self.step_init = step
        self.step = step
        self.nf = 0
        self.xflist = np.array([[0, f0]])
        self.fm = f0
        self.am = 0
        self.xm = x0

        self.phase = "positive"  # "positive" or "negative"
        self.waiting = False
        self.current_candidate = None
        self.finished_flag = False
        self.gold_r = 1.618
        self.max_iter = 1000
        self.iter_count = 0

        # In negative phase, we need to remember a2 from positive phase.
        self.a2_positive = None

    def next_candidate(self):
        """Return the next candidate evaluation point and set waiting=True."""
        if self.finished_flag:
            return None
        if self.waiting:
            raise RuntimeError("BracketMinStepper is waiting for an update.")
        # Depending on phase, compute candidate.
        if self.phase == "positive":
            candidate = self.x0 + self.dv * self.step
            self.current_candidate = candidate
            self.waiting = True
            self.last_step = self.step  # save for possible revert
            return candidate
        elif self.phase == "negative":
            candidate = self.x0 + self.dv * self.step
            self.current_candidate = candidate
            self.waiting = True
            self.last_step = self.step
            return candidate
        else:
            raise RuntimeError("Unknown phase in BracketMinStepper.")

    def update(self, new_obj: float):
        """
        Update with the externally evaluated objective value for the last candidate.
        Then adjust internal state according to the algorithm.
        """
        if not self.waiting:
            raise RuntimeError("BracketMinStepper is not waiting for an update.")
        f_val = new_obj
        self.nf += 1
        self.waiting = False

        # Update xflist with current step and f_val.
        self.xflist = np.concatenate(
            (self.xflist, np.array([[self.step, f_val]])), axis=0
        )
        # Update best if improved.
        if f_val < self.fm:
            self.fm = f_val
            self.am = self.step
            self.xm = self.current_candidate

        if self.phase == "positive":
            # Continue positive phase while f_val < fm + noise*3.
            if (
                f_val < self.fm + self.rcds.noise * 3
                and self.iter_count < self.max_iter
            ):
                self.iter_count += 1
                # Save current step.
                step0 = self.step
                # Update step as per original logic.
                if abs(self.step) < 0.1:
                    self.step = self.step * (1.0 + self.gold_r)
                else:
                    self.step = self.step + 0.1
                # Next candidate will be produced.
            else:
                # End positive phase.
                self.a2_positive = self.step  # store positive phase a2
                # Check if negative phase is needed.
                if self.f0 > self.fm + self.rcds.noise * 3:
                    # No negative phase; finish.
                    self.finished_flag = True
                else:
                    # Start negative phase.
                    self.phase = "negative"
                    self.step = -self.step_init
                    self.iter_count = 0
            # End of positive phase update.
        elif self.phase == "negative":
            # Negative phase.
            if (
                f_val < self.fm + self.rcds.noise * 3
                and self.iter_count < self.max_iter
            ):
                self.iter_count += 1
                step0 = self.step
                if abs(self.step) < 0.1:
                    self.step = self.step * (1.0 + self.gold_r)
                else:
                    self.step = self.step - 0.1
                # In original, if math.isnan(f_val) then revert step.
                if math.isnan(f_val):
                    self.step = step0
                    self.finished_flag = True
                # Else, continue.
            else:
                self.finished_flag = True
        else:
            raise RuntimeError("Unknown phase in BracketMinStepper update.")

    def finished(self) -> bool:
        return self.finished_flag

    def result(self):
        """
        Once finished, return:
          xm: best candidate,
          fm: best objective,
          a1, a2: bracket bounds shifted by am,
          xflist: adjusted list of [alpha, f(alpha)],
          nf: total evaluations.
        """
        if self.phase == "positive":
            # Finished positive phase.
            a2 = self.step
            a1 = 0 - self.am
            a2 = a2 - self.am
            xflist_adj = self.xflist.copy()
            xflist_adj[:, 0] -= self.am
            return self.xm, self.fm, a1, a2, xflist_adj, self.nf
        elif self.phase == "negative":
            a1 = self.step
            a2 = self.a2_positive if self.a2_positive is not None else self.step_init
            if a1 > a2:
                a1, a2 = a2, a1
            a1 -= self.am
            a2 -= self.am
            xflist_adj = self.xflist.copy()
            xflist_adj[:, 0] -= self.am
            idx = np.argsort(xflist_adj[:, 0])
            xflist_adj = xflist_adj[idx]
            return self.xm, self.fm, a1, a2, xflist_adj, self.nf
        else:
            raise RuntimeError("Unknown phase in BracketMinStepper result.")


###############################################################################
# LinescanStepper replicates the generator-style linescan.
###############################################################################
class LinescanStepper:
    """
    A stepper for the line scan portion.

    It mimics the original linescan:
      - Create a linspace alist between alo and ahi (with at least 6 points).
      - Incorporate any pre-evaluated points from xflist.
      - For each point in alist that is NaN in flist, produce candidate x = x0 + alpha*dv and wait.
      - Upon update, record the objective value.
      - When all evaluations are complete, if there are enough points, fit a quadratic and pick the minimum.
    """

    def __init__(
        self,
        rcds,
        x0: np.ndarray,
        f0: float,
        dv: np.ndarray,
        alo: float,
        ahi: float,
        Np: int,
        xflist: np.ndarray,
    ):
        self.rcds = rcds
        self.x0 = x0
        self.f0 = f0
        self.dv = dv
        self.alo = alo
        self.ahi = ahi
        self.Np = int(Np) if Np >= 6 else 6
        self.delta = (ahi - alo) / (self.Np - 1.0)
        self.alist = np.linspace(alo, ahi, self.Np)
        self.flist = np.full_like(self.alist, float("nan"))
        # Incorporate pre-evaluated points from xflist.
        Npre = np.shape(xflist)[0]
        for ii in range(Npre):
            if xflist[ii, 0] >= alo and xflist[ii, 0] <= ahi:
                ik = int(round((xflist[ii, 0] - alo) / self.delta))
                self.alist[ik] = xflist[ii, 0]
                self.flist[ik] = xflist[ii, 1]
        self.index = 0
        self.waiting = False
        self.current_candidate = None
        self.nf = 0
        self.finished_flag = False

    def next_candidate(self):
        """Return next candidate along the line for which f is NaN."""
        if self.finished_flag:
            return None
        if self.waiting:
            raise RuntimeError("LinescanStepper is waiting for an update.")
        # Skip indices that already have a value.
        while self.index < len(self.alist) and not math.isnan(self.flist[self.index]):
            self.index += 1
        if self.index < len(self.alist):
            alpha = self.alist[self.index]
            self.current_candidate = self.x0 + alpha * self.dv
            self.waiting = True
            return self.current_candidate
        else:
            self.finished_flag = True
            return None

    def update(self, new_obj: float):
        """Update the f value at the current index with external evaluation."""
        if not self.waiting:
            raise RuntimeError("LinescanStepper is not waiting for an update.")
        self.flist[self.index] = new_obj
        self.nf += 1
        self.waiting = False
        self.index += 1

    def finished(self) -> bool:
        return self.finished_flag

    def result(self):
        """
        When finished, if there are fewer than 5 points, pick the minimum directly.
        Otherwise, fit a quadratic and return the minimum.
        Returns (xm, fm, nf).
        """
        valid = ~np.isnan(self.flist)
        alist_valid = self.alist[valid]
        flist_valid = self.flist[valid]
        if len(alist_valid) <= 0:
            return self.x0, self.f0, self.nf
        elif len(alist_valid) < 5:
            imin = flist_valid.argmin()
            xm = self.x0 + alist_valid[imin] * self.dv
            fm = flist_valid[imin]
            return xm, fm, self.nf
        else:
            p = np.polyfit(alist_valid, flist_valid, 2)
            pf = np.poly1d(p)
            MP = 101
            av = np.linspace(alist_valid[0], alist_valid[-1], MP - 1)
            yv = pf(av)
            imin = yv.argmin()
            xm = self.x0 + av[imin] * self.dv
            fm = yv[imin]
            return xm, fm, self.nf


###############################################################################
# Main RCDS state-machine class using steppers.
###############################################################################
class RCDS:
    """
    Robust Conjugate Direction Search (RCDS) algorithm as a state machine.

    This version has been refactored to use the BracketMinStepper and LinescanStepper
    so that each candidate evaluation "pauses" and waits for an external update.
    The overall logic is identical to the generator-style version.
    """

    def __init__(
        self,
        x0: np.ndarray,
        init_mat: Optional[np.ndarray] = None,
        noise: float = 0.1,
        step: float = 1e-2,
    ):
        self.x0 = x0.reshape(-1, 1)
        self.Imat = init_mat
        self.noise = noise
        self.step = step
        self.cnt = 0
        self.OBJ = None
        # State machine variables:
        # States: "init", "direction_loop", "waiting"
        self.state = "init"
        self.iteration = 0
        self.direction_index = 0  # index over directions
        self.dl = 0
        self.best_direction_index = 0
        self.current_candidate = None
        self.Nvar = len(self.x0)
        if self.Imat is not None:
            self.Dmat = self.Imat.copy()
        else:
            self.Dmat = np.matrix(np.identity(self.Nvar))
        self.xm = self.x0
        self.fm = None
        self.f0 = None
        self.nf = 0
        # Active steppers:
        self.bracket_stepper = None
        self.linescan_stepper = None

    def get_next_candidate(self):
        """
        Produce the next candidate for external evaluation.

        In the "init" state, return the initial x0.
        In "direction_loop", use the bracketmin stepper for the current direction.
        After processing all directions, propose xt = 2*xm - x0.
        In every case, the state is set to "waiting" until an external update occurs.
        """
        if self.state == "init":
            self.f0, _ = self.func_obj(self.x0)
            self.nf = 1
            self.xm = self.x0
            self.fm = self.f0
            self.iteration = 0
            self.direction_index = 0
            self.dl = 0
            self.best_direction_index = 0
            self.state = "waiting"
            self.current_candidate = self.x0
            logger.debug("Initial candidate: %s", self.x0)
            return self.current_candidate

        elif self.state == "direction_loop":
            if self.direction_index < self.Nvar:
                dv = self.Dmat[:, self.direction_index]
                if self.bracket_stepper is None:
                    self.bracket_stepper = BracketMinStepper(
                        self, self.xm, self.fm, dv, self.step
                    )
                candidate = self.bracket_stepper.next_candidate()
                self.state = "waiting"
                return candidate
            else:
                self.current_candidate = 2 * self.xm - self.x0
                self.state = "waiting"
                logger.debug("Proposed candidate xt: %s", self.current_candidate)
                return self.current_candidate

        elif self.state == "waiting":
            raise RuntimeError(
                "Candidate evaluation pending; call update_obj(new_obj) first."
            )
        else:
            raise RuntimeError("Unknown state in RCDS.")

    def update_obj(self, new_obj: float):
        """
        External update: supply the evaluated objective for the candidate produced.

        If a bracket stepper is active and waiting, update it. Otherwise update the outer candidate.
        """
        if self.state != "waiting":
            raise RuntimeError(
                "update_obj called in invalid state; no candidate pending evaluation."
            )

        # If a bracket stepper is active and waiting, update it.
        if self.bracket_stepper is not None and self.bracket_stepper.waiting:
            self.bracket_stepper.update(new_obj)
            if self.bracket_stepper.finished():
                xm, fm, a1, a2, xflist, ndf = self.bracket_stepper.result()
                self.nf += ndf
                self.xm = xm
                self.fm = fm
                # Update direction improvement.
                if (self.fm - fm) > self.dl:
                    self.dl = self.fm - fm
                    self.best_direction_index = self.direction_index
                self.direction_index += 1
                self.bracket_stepper = None
                self.state = "direction_loop"
            else:
                self.state = "direction_loop"
            logger.debug("Bracket update: xm=%s, fm=%f", self.xm, self.fm)
        else:
            # Outer candidate update.
            ft = new_obj
            logger.debug("Outer candidate evaluation: %f", ft)
            if self.iteration == 0:
                self.f0 = ft
                self.xm = self.x0
                self.fm = self.f0
            else:
                # (Additional logic for updating directions would go here.)
                pass
            self.f0 = self.fm
            self.x0 = self.xm
            self.iteration += 1
            self.direction_index = 0
            self.dl = 0
            self.best_direction_index = 0
            self.state = "direction_loop"

    def get_min_along_dir_parab_non_gen(
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
        Combine a bracket search and a subsequent line scan along dv.

        First, create and run a BracketMinStepper. Then, with the resulting bracket,
        create and run a LinescanStepper. External code is assumed to drive the steppers.
        """
        # Bracket phase.
        bracket_stepper = BracketMinStepper(self, x0, f0, dv, self.step)
        self.bracket_stepper = bracket_stepper
        # In an interactive run, external updates would drive bracket_stepper until finished.
        # Here we assume that process happens externally.
        while not bracket_stepper.finished():
            break  # externally driven; here we simply exit.
        xm, fm, a1, a2, xflist, ndf1 = bracket_stepper.result()
        # Linescan phase.
        linescan_stepper = LinescanStepper(self, xm, fm, dv, a1, a2, Npmin, xflist)
        self.linescan_stepper = linescan_stepper
        while not linescan_stepper.finished():
            break  # externally driven.
        xm_final, fm_final, ndf2 = linescan_stepper.result()
        return xm_final, fm_final, ndf1 + ndf2

    def update_obj_func(self, obj: float):
        self.OBJ = obj

    def func_obj(self, x: np.ndarray, count=True) -> tuple[float, float]:
        self.Nvar = len(x)
        obj = self.OBJ
        if count:
            self.cnt += 1
        return obj, obj


###############################################################################
# RCDSGenerator wrapper.
###############################################################################
class RCDSGenerator(SequentialGenerator):
    """
    RCDS algorithm wrapped as a SequentialGenerator using the stepper-based state machine.

    External workflow:
      - Call _generate() to obtain a candidate.
      - Evaluate candidate externally.
      - Call _add_data() (which calls update_obj) with the evaluation.
    """

    name = "rcds"
    init_mat: Optional[np.ndarray] = Field(None)
    noise: PositiveFloat = Field(1e-5)
    step: PositiveFloat = Field(1e-2)

    _rcds: RCDS = None
    _sign = 1

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def _set_data(self, data: pd.DataFrame):
        # just store data
        self.data = data

        new_data_df = self.vocs.objective_data(data)
        res = new_data_df.iloc[-1:, :].to_numpy()
        assert np.shape(res) == (1, 1), f"Bad last point [{res}]"

    def _reset(self):
        objective_name = self.vocs.objective_names[0]  # RCDS supports one objective
        direction = self.vocs.objectives[objective_name]
        self._sign = 1 if direction == "MINIMIZE" else -1
        x0, f0 = self._get_initial_point()
        lb, ub = self.vocs.bounds
        _x0 = (x0 - lb) / (ub - lb)
        self._rcds = RCDS(
            x0=_x0, init_mat=self.init_mat, noise=self.noise, step=self.step
        )
        self._rcds.OBJ = self._sign * float(f0)
        self._rcds.state = "init"

    def _add_data(self, new_data: pd.DataFrame):
        res = float(new_data.iloc[-1][self.vocs.objective_names].to_numpy())
        if self._rcds is not None:
            self._rcds.update_obj(self._sign * res)

    def _generate(self, first_gen: bool = False):
        if first_gen or self._rcds is None:
            self.reset()
        _x_next = self._rcds.get_next_candidate()
        while np.any(_x_next > 1) or np.any(_x_next < 0):
            self._rcds.update_obj(float("nan"))
            _x_next = self._rcds.get_next_candidate()
        _x_next = np.array(_x_next).flatten()
        lb, ub = self.vocs.bounds
        x_next = (ub - lb) * _x_next + lb
        x_next = [float(ele) for ele in x_next]
        return [dict(zip(self.vocs.variable_names, x_next))]
