import logging
import math
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field
from pydantic.types import PositiveFloat

from xopt.generators.sequential.sequential_generator import SequentialGenerator

logger = logging.getLogger(__name__)


class StateMachineFinishedError(Exception):
    """Raised when the state machine has finished. Contains the result tuple."""

    def __init__(self, result):
        self.result = result
        super().__init__(f"State machine finished with result: {result}")


class BracketMinStateMachine:
    def __init__(self, noise, x0: np.ndarray, f0: float, dv: np.ndarray, step: float):
        """
        Initialize with parameters and OBJ.
        """
        self.noise = noise
        self.x0 = x0
        self.f0 = f0
        self.dv = dv
        self.step = step
        self.step_init = step
        self.gold_r = 1.618

        self.nf = 0
        # List of [alpha, f(alpha)] values.
        self.xflist = [[0, f0]]
        self.fm = f0
        self.am = 0
        self.xm = x0
        self.a2 = None  # to be set after forward phase
        self.last_step = None
        self.result = None

        # flag to ensure candidate update order
        self.pending = False
        self.current_branch = None

        # Determine initial phase: if f0 is NaN, we propose x0 first.
        if math.isnan(f0):
            self.phase = "initial_nan_wait"
        else:
            self.phase = "forward_first"

    def propose(self):
        """
        Propose the next candidate.
        If finished, raises an error containing the final result.
        """
        if self.phase == "finished":
            raise StateMachineFinishedError(self.result)
        if self.pending:
            raise Exception(
                "A candidate is already pending; please call update_obj() first."
            )

        # --- INITIAL NAN CASE ---
        if self.phase == "initial_nan_wait":
            candidate = self.x0
            self.pending = True
            self.current_branch = "initial_nan"
            return candidate

        # --- FORWARD PHASE: first candidate ---
        elif self.phase == "forward_first":
            candidate = self.x0 + self.dv * self.step
            self.pending = True
            self.current_branch = "forward"
            self.phase = "forward_wait"
            return candidate

        # --- FORWARD PHASE: subsequent candidates ---
        elif self.phase == "forward_loop":
            self.last_step = self.step
            if abs(self.step) < 0.1:
                self.step = self.step * (1.0 + self.gold_r)
            else:
                self.step = self.step + 0.1
            candidate = self.x0 + self.dv * self.step
            self.pending = True
            self.current_branch = "forward"
            self.phase = "forward_wait"
            return candidate

        # --- POST-FORWARD PHASE: decide on negative search or finish ---
        elif self.phase == "post_forward":
            self.a2 = self.step
            if self.f0 > self.fm + self.noise * 3:
                a1 = 0 - self.am
                a2 = self.a2 - self.am
                arr = np.array(self.xflist, dtype=float)
                arr[:, 0] -= self.am
                self.result = (self.xm, self.fm, a1, a2, arr, self.nf)
                self.phase = "finished"
                raise StateMachineFinishedError(self.result)
            else:
                self.step = -self.step_init
                candidate = self.x0 + self.dv * self.step
                self.pending = True
                self.current_branch = "negative"
                self.phase = "negative_wait"
                return candidate

        # --- NEGATIVE PHASE: subsequent candidates ---
        elif self.phase == "negative_loop":
            self.last_step = self.step
            if abs(self.step) < 0.1:
                self.step = self.step * (1.0 + self.gold_r)
            else:
                self.step = self.step - 0.1
            candidate = self.x0 + self.dv * self.step
            self.pending = True
            self.current_branch = "negative"
            self.phase = "negative_wait"
            return candidate

        # --- FINALIZE NEGATIVE PHASE ---
        elif self.phase == "finalize_negative":
            a1 = self.step  # current negative step
            a2 = self.a2  # a2 was set in the forward phase
            if a1 > a2:
                a1, a2 = a2, a1
            a1 -= self.am
            a2 -= self.am
            arr = np.array(self.xflist, dtype=float)
            arr[:, 0] -= self.am
            arr = arr[np.argsort(arr[:, 0])]
            self.result = (self.xm, self.fm, a1, a2, arr, self.nf)
            self.phase = "finished"
            raise StateMachineFinishedError(self.result)

        else:
            raise Exception("Invalid phase in propose: " + self.phase)

    def update_obj(self, obj):
        """
        Update the state machine with the evaluated objective value.
        Expects obj to be a tuple like (f_value, ...).
        """
        if not self.pending:
            raise Exception("No candidate pending update.")
        f_val = obj
        self.pending = False

        if self.current_branch == "initial_nan":
            self.nf += 1  # Count this evaluation to match the generator.
            self.f0 = f_val
            self.xflist[0][1] = f_val
            self.fm = f_val
            self.phase = "forward_first"

        elif self.current_branch == "forward":
            self.nf += 1
            self.xflist.append([self.step, f_val])
            if f_val < self.fm:
                self.fm = f_val
                self.am = self.step
                self.xm = self.x0 + self.dv * self.step
            if math.isnan(f_val):
                if self.last_step is not None:
                    self.step = self.last_step
                self.phase = "post_forward"
            else:
                if f_val < self.fm + self.noise * 3:
                    self.phase = "forward_loop"
                else:
                    self.phase = "post_forward"

        elif self.current_branch == "negative":
            self.nf += 1
            self.xflist.append([self.step, f_val])
            if f_val < self.fm:
                self.fm = f_val
                self.am = self.step
                self.xm = self.x0 + self.dv * self.step
            if math.isnan(f_val):
                if self.last_step is not None:
                    self.step = self.last_step
                self.phase = "finalize_negative"
            else:
                if f_val < self.fm + self.noise * 3:
                    self.phase = "negative_loop"
                else:
                    self.phase = "finalize_negative"
        else:
            raise Exception("Invalid branch in update_obj: " + self.current_branch)


class LineScanStateMachine:
    def __init__(
        self,
        x0: np.ndarray,
        f0: float,
        dv: np.ndarray,
        alo: float,
        ahi: float,
        Np: int,
        xflist: np.ndarray,
    ):
        """
        Initialize the state machine with parameters.

        OBJ is a member variable that is updated externally by assignment.
        """
        self.x0 = x0
        self.f0 = f0
        self.dv = dv
        self.alo = alo
        self.ahi = ahi
        self.Np = Np
        self.xflist_input = xflist  # extra data from previous evaluations

        self.nf = 0  # evaluation counter

        # Phases:
        # "initial_nan_wait": waiting to evaluate x0 because f0 is NaN.
        # "setup": perform error checks and compute alist/flist.
        # "linescan_loop": loop through indices where evaluation is missing.
        # "waiting_evaluation": a candidate has been proposed and we await its evaluation.
        # "finalize": process all evaluations and compute the final candidate.
        # "finished": final result is available.
        if math.isnan(f0):
            self.phase = "initial_nan_wait"
        else:
            self.phase = "setup"

        self.pending = False  # flag that a candidate is waiting evaluation
        self.pending_index = None  # for "linescan_loop" branch
        self.current_branch = None  # distinguishes the initial_nan branch

        # These will be computed during setup.
        self.alist = None
        self.flist = None
        self.delta = None
        self.current_index = 0  # pointer in the alist loop

        self.result = None  # final result will be stored here

    def propose(self):
        """
        Propose the next candidate point.
        If the process is complete, raise StateMachineFinishedError containing the final result.
        """
        if self.phase == "finished":
            raise StateMachineFinishedError(self.result)
        if self.pending:
            raise Exception("A candidate is already pending; call update_obj() first.")

        # === INITIAL NAN PHASE ===
        if self.phase == "initial_nan_wait":
            self.pending = True
            self.current_branch = "initial_nan"
            return self.x0

        # === SETUP PHASE: perform error checking and initialize alist, flist ===
        if self.phase == "setup":
            # Check for errors.
            if self.alo >= self.ahi:
                print("Error: bracket upper bound equal to or lower than lower bound")
                self.result = (self.x0, self.f0, self.nf)
                self.phase = "finished"
                raise StateMachineFinishedError(self.result)
            if len(self.x0) != len(self.dv):
                print("Error: x0 and dv dimension do not match.")
                self.result = (self.x0, self.f0, self.nf)
                self.phase = "finished"
                raise StateMachineFinishedError(self.result)

            # Adjust Np if needed.
            if math.isnan(self.Np) or self.Np < 6:
                self.Np = 6
            self.delta = (self.ahi - self.alo) / (self.Np - 1.0)
            # Create alist: a linear space between alo and ahi.
            self.alist = np.linspace(self.alo, self.ahi, self.Np)
            # Create flist: same shape, all values NaN.
            self.flist = np.full_like(self.alist, float("nan"))

            # Incorporate previous evaluations from xflist_input.
            Nlist = np.shape(self.xflist_input)[0]
            for ii in range(Nlist):
                # If the stored evaluation is within bounds...
                if (
                    self.xflist_input[ii, 1] >= self.alo
                    and self.xflist_input[ii, 1] <= self.ahi
                ):
                    ik = round((self.xflist_input[ii, 1] - self.alo) / self.delta)
                    self.alist[ik] = self.xflist_input[ii, 0]
                    self.flist[ik] = self.xflist_input[ii, 1]

            self.current_index = 0  # start processing alist from index 0
            self.phase = "linescan_loop"
            return self.propose()  # immediately continue to next phase

        # === LOOP PHASE: propose candidate for missing evaluations ===
        if self.phase == "linescan_loop":
            if self.current_index < len(self.alist):
                # Check if the current candidate has not been evaluated.
                if math.isnan(self.flist[self.current_index]):
                    # Candidate needs evaluation.
                    candidate = self.x0 + self.alist[self.current_index] * self.dv
                    self.pending = True
                    # Set pending_index to current_index before incrementing.
                    self.pending_index = self.current_index
                    self.current_index += 1
                    self.phase = "waiting_evaluation"
                    return candidate
                else:
                    # Already evaluated; move on.
                    self.current_index += 1
                    return self.propose()
            else:
                # All indices processed; move to final processing.
                self.phase = "finalize"
                return self.propose()

        # === FINALIZE PHASE: process evaluations and compute final candidate ===
        if self.phase == "finalize":
            # Build a mask for valid evaluations.
            mask = ~np.isnan(self.flist)
            alist_valid = self.alist[mask]
            flist_valid = self.flist[mask]
            if len(alist_valid) <= 0:
                self.result = (self.x0, self.f0, self.nf)
            elif len(alist_valid) < 5:
                imin = flist_valid.argmin()
                xm = self.x0 + alist_valid[imin] * self.dv
                fm = flist_valid[imin]
                self.result = (xm, fm, self.nf)
            else:
                # Use a quadratic fit.
                p = np.polyfit(alist_valid, flist_valid, 2)
                pf = np.poly1d(p)
                MP = 101
                av = np.linspace(alist_valid[0], alist_valid[-1], MP - 1)
                yv = pf(av)
                imin = yv.argmin()
                xm = self.x0 + av[imin] * self.dv
                fm = yv[imin]
                self.result = (xm, fm, self.nf)
            self.phase = "finished"
            raise StateMachineFinishedError(self.result)

        raise Exception("Invalid phase in propose: " + self.phase)

    def update_obj(self, obj):
        """
        Update the state machine with the evaluated objective value.

        For this routine, the external code directly sets the member OBJ (via assignment)
        so that here the parameter `obj` is simply the numeric evaluation.
        """
        if not self.pending:
            raise Exception("No candidate pending update.")
        f_val = obj  # the evaluated value

        # === INITIAL NAN UPDATE ===
        if self.current_branch == "initial_nan":
            self.nf += 1
            self.f0 = f_val
            self.phase = "setup"
            self.pending = False
            self.current_branch = None
            return

        # === UPDATE DURING LINESCAN LOOP ===
        if self.phase == "waiting_evaluation":
            self.flist[self.pending_index] = f_val
            self.nf += 1
            self.pending = False
            self.phase = "linescan_loop"
            return

        raise Exception("Invalid phase in update_obj: " + self.phase)


class GetMinAlongDirParabStateMachine:
    """
    State machine that combines a bracketmin phase and a subsequent linescan phase.

    The process is:
      1. Create and run a BracketMinStateMachine with (x0, f0, dv, step).
         It will yield candidate points until it finishes with a result:
             (x1, f1, a1, a2, xflist, ndf1)
      2. Using that result, create and run a LineScanStateMachine with parameters
         (x1, f1, dv, a1, a2, Npmin, xflist). It yields candidate points until finished,
         with result (x1, f1, ndf2).
      3. The final result of this combined process is (x1, f1, ndf1+ndf2).

    Use propose() to get a candidate and update_obj() to send back the evaluated value.
    When finished, propose() raises StateMachineFinishedError with the final result.
    """

    def __init__(
        self,
        x0: np.ndarray,
        f0: float,
        dv: np.ndarray,
        Npmin: int = 6,
        step: float = 1.0,
        it: int = None,
        idx: int = None,
        replaced: bool = False,
        noise: float = 0.1,
    ):
        self.x0 = x0
        self.f0 = f0
        self.dv = dv
        self.Npmin = Npmin
        self.step = step
        # it, idx, replaced can be stored or used as needed.
        self.it = it
        self.idx = idx
        self.replaced = replaced
        self.noise = noise

        # We initialize the bracketmin phase first.
        # For the OBJ member variable, we use a dummy initial value (NaN); external code
        # is expected to evaluate candidates and then update this value.
        self.bm = BracketMinStateMachine(
            self.noise, self.x0, self.f0, self.dv, self.step
        )
        self.ls = None  # Will hold the LineScanStateMachine instance later.
        self.phase = "bracketmin"  # Current phase: "bracketmin", then "linescan", then "finished".
        self.ndf1 = None  # To store function eval count from bracketmin.
        self.result = None  # Final result: (x1, f1, ndf1 + ndf2).
        self.pending = False  # Whether a candidate is waiting an update.

    def propose(self):
        """
        Propose the next candidate.
        Delegates to the current active submachine (bracketmin or linescan).
        When the process is complete, raises StateMachineFinishedError with final result.
        """
        if self.phase == "finished":
            raise StateMachineFinishedError(self.result)
        if self.pending:
            raise Exception(
                "A candidate is already pending evaluation; please call update_obj() first."
            )

        # --- Phase 1: BracketMin ---
        if self.phase == "bracketmin":
            try:
                candidate = self.bm.propose()
                self.pending = True
                return candidate
            except StateMachineFinishedError as e:
                # BracketMin is finished; extract its final result.
                # Expected result: (x1, f1, a1, a2, xflist, ndf1)
                x1, f1, a1, a2, xflist, ndf1 = e.result
                self.ndf1 = ndf1
                # Now start the linescan phase with these values.
                self.ls = LineScanStateMachine(
                    x1, f1, self.dv, a1, a2, self.Npmin, xflist
                )
                self.phase = "linescan"
                # Immediately delegate to the linescan machine.
                return self.propose()

        # --- Phase 2: LineScan ---
        if self.phase == "linescan":
            try:
                candidate = self.ls.propose()
                self.pending = True
                return candidate
            except StateMachineFinishedError as e:
                # Linescan finished; result is (x1, f1, ndf2)
                x1, f1, ndf2 = e.result
                total_ndf = self.ndf1 + ndf2
                self.result = (x1, f1, total_ndf)
                self.phase = "finished"
                raise StateMachineFinishedError(self.result)

        raise Exception(
            "Invalid phase in GetMinAlongDirParabStateMachine: " + self.phase
        )

    def update_obj(self, obj):
        """
        Update the state machine with the evaluated objective value.
        Delegates to the currently active submachine.
        """
        if not self.pending:
            raise Exception("No candidate pending update.")
        # Clear the pending flag and delegate the update.
        self.pending = False
        if self.phase == "bracketmin":
            self.bm.update_obj(obj)
        elif self.phase == "linescan":
            self.ls.update_obj(obj)
        else:
            raise Exception("Invalid phase for update_obj: " + self.phase)


class PowellMainStateMachine:
    def __init__(self, x0: np.ndarray, step: float, Imat=None, noise=0.1):
        """
        Initialize with:
          - x0: starting point (np.ndarray)
          - step: initial step size
          - Imat: optional initial direction matrix; if None, the identity is used.
        """
        self.x0 = x0.copy()  # previous iteration's starting point
        self.step = step
        self.Nvar = len(x0)
        self.Imat = Imat
        self.noise = noise
        # State variables:
        self.phase = (
            "init"  # phases: "init", "iteration_start", "line_search", "extrapolation",
        )
        # "direction_update", "iteration_end", "finished"
        self.pending = False  # True if a candidate is waiting evaluation update
        self.nf = 0  # total function evaluation counter
        self.iteration = 0  # iteration counter
        self.x_current = None  # current best point (xm)
        self.f_current = None  # current best function value (fm)
        self.Dmat = None  # direction matrix
        # For inner loop over directions:
        self.inner_index = 0  # which coordinate (column) we are scanning
        self.inner_dl = 0  # best improvement (delta) in this iteration
        self.inner_k = 0  # index of direction giving maximum improvement
        self.current_gmadp = None  # current GetMinAlongDirParab state machine instance
        # For extrapolation phase:
        self.xt = None  # extrapolated candidate

    def propose(self):
        if self.phase == "finished":
            raise StateMachineFinishedError((self.x_current, self.f_current, self.nf))
        if self.pending:
            raise Exception("Candidate pending evaluation; call update_obj() first.")

        # --- INIT PHASE: Yield the starting point x0 ---
        if self.phase == "init":
            self.pending = True
            self.phase = "init_wait"
            return self.x0

        # --- ITERATION START: Setup a new iteration ---
        if self.phase == "iteration_start":
            self.iteration += 1
            # Reset inner loop counters:
            self.inner_index = 0
            self.inner_dl = 0
            self.inner_k = 0
            self.current_gmadp = None
            self.phase = "line_search"
            return self.propose()

        # --- LINE SEARCH: Loop over each direction in Dmat ---
        if self.phase == "line_search":
            if self.inner_index < self.Nvar:
                # If no current submachine, create one for the current direction.
                if self.current_gmadp is None:
                    # dv is the current search direction: column inner_index of Dmat.
                    dv = self.Dmat[:, self.inner_index]
                    # Create the get_min_along_dir_parab state machine.
                    self.current_gmadp = GetMinAlongDirParabStateMachine(
                        self.x_current,
                        self.f_current,
                        dv,
                        Npmin=6,
                        step=self.step,
                        it=self.iteration,
                        idx=self.inner_index,
                        noise=self.noise,
                    )
                try:
                    candidate = self.current_gmadp.propose()
                    self.pending = True
                    return candidate
                except StateMachineFinishedError as e:
                    # Submachine finished; unpack its result: (x1, f1, ndf)
                    x1, f1, ndf = e.result
                    self.nf += ndf
                    # Update best improvement if achieved.
                    if (self.f_current - f1) > self.inner_dl:
                        self.inner_dl = self.f_current - f1
                        self.inner_k = self.inner_index
                    # Update current best:
                    self.x_current = x1
                    self.f_current = f1
                    self.current_gmadp = None
                    self.inner_index += 1
                    return self.propose()
            else:
                # Finished scanning all coordinate directions.
                # Move to extrapolation phase.
                self.xt = 2 * self.x_current - self.x0
                self.pending = True
                self.phase = "extrapolation"
                return self.xt

        # --- EXTRAPOLATION PHASE: Yield extrapolated candidate xt ---
        if self.phase == "extrapolation":
            raise Exception("Waiting for evaluation update of extrapolated candidate.")

        # --- DIRECTION UPDATE PHASE: Possibly update Dmat and search along new direction ---
        if self.phase == "direction_update":
            if self.current_gmadp is None:
                # Compute new direction ndv from (x_current - x0)
                diff = self.x_current - self.x0
                norm_diff = np.linalg.norm(diff)
                ndv = diff / norm_diff if norm_diff != 0 else diff
                # Compute dot products of ndv with each column of Dmat.
                dotp = np.array(
                    [abs(np.dot(ndv.T, self.Dmat[:, j])) for j in range(self.Nvar)]
                )
                if max(dotp) < 0.9:
                    # Replace the direction corresponding to inner_k:
                    for j in range(self.inner_k, self.Nvar - 1):
                        self.Dmat[:, j] = self.Dmat[:, j + 1]
                    self.Dmat[:, -1] = ndv
                    dv = self.Dmat[:, -1]
                    self.current_gmadp = GetMinAlongDirParabStateMachine(
                        self.x_current,
                        self.f_current,
                        dv,
                        Npmin=6,
                        step=self.step,
                        it=self.iteration,
                        idx=self.inner_index,
                        noise=self.noise,
                    )
                else:
                    # No new direction update; skip to iteration end.
                    self.phase = "iteration_end"
                    return self.propose()
            try:
                candidate = self.current_gmadp.propose()
                self.pending = True
                return candidate
            except StateMachineFinishedError as e:
                x1, f1, ndf = e.result
                self.nf += ndf
                self.x_current = x1
                self.f_current = f1
                self.current_gmadp = None
                self.phase = "iteration_end"
                return self.propose()

        # --- ITERATION END: Update for next iteration ---
        if self.phase == "iteration_end":
            self.x0 = self.x_current.copy()
            self.f0 = self.f_current
            self.phase = "iteration_start"
            return self.propose()

        raise Exception("Unknown phase in propose: " + self.phase)

    def update_obj(self, obj):
        """
        Update the state machine with the evaluated objective value.
        The parameter 'obj' is the evaluated function value.
        """
        if not self.pending:
            raise Exception("No candidate pending update.")
        self.pending = False
        # --- Update for initial candidate ---
        if self.phase == "init_wait":
            self.f0 = obj
            self.nf += 1
            self.x_current = self.x0.copy()
            self.f_current = self.f0
            # Initialize Dmat: use self.Imat if provided; otherwise identity.
            if self.Imat is not None:
                self.Dmat = self.Imat.copy()
            else:
                self.Dmat = np.array(np.identity(self.Nvar))
            self.phase = "iteration_start"
            return
        # --- Delegate update to the active inner state machine ---
        if self.phase == "line_search":
            # Forward the update to the current get_min_along_dir_parab state machine.
            self.current_gmadp.update_obj(obj)
            return
        # --- Update for extrapolated candidate xt ---
        if self.phase == "extrapolation":
            self.ft = obj  # ft = f(xt)
            self.nf += 1
            # Test the condition from the original code:
            # If f0 <= ft or 2*(f0-2*fm+ft)*((f0-fm-dl)/(ft-f0))**2 >= dl then no new direction update.
            if (
                self.f0 <= self.ft
                or 2
                * (self.f0 - 2 * self.f_current + self.ft)
                * ((self.f0 - self.f_current - self.inner_dl) / (self.ft - self.f0))
                ** 2
                >= self.inner_dl
            ):
                self.phase = "iteration_end"
            else:
                self.phase = "direction_update"
            return
        # --- Delegate update during direction update phase ---
        if self.phase == "direction_update":
            self.current_gmadp.update_obj(obj)
            return

        raise Exception("Invalid phase in update_obj: " + self.phase)


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
    _ub : np.ndarray
        Upper bounds of the variables.
    _lb : np.ndarray
        Lower bounds of the variables.
    _powell : PowellMainStateMachine
        Instance of the PowellMainStateMachine.
    _sign : int
        Sign of the objective function (1 for MINIMIZE, -1 for MAXIMIZE).
    """

    name = "rcds"
    supports_single_objective: bool = True
    init_mat: Optional[np.ndarray] = Field(None)
    noise: PositiveFloat = Field(1e-5)
    step: PositiveFloat = Field(1e-2)

    _powell: PowellMainStateMachine = None
    _sign = 1

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def _reset(self):
        """reset the powell object"""

        objective_name = self.vocs.objective_names[
            0
        ]  # rcds only supports one objective
        direction = self.vocs.objectives[objective_name]
        if direction == "MINIMIZE":
            self._sign = 1
        elif direction == "MAXIMIZE":
            self._sign = -1

        x0, f0 = self._get_initial_point()

        # RCDS assume a normalized problem
        lb, ub = self.vocs.bounds
        _x0 = (x0 - lb) / (ub - lb)

        self._powell = PowellMainStateMachine(
            x0=_x0,
            step=self.step,
            Imat=self.init_mat,
            noise=self.noise,
        )
        _ = self._powell.propose()
        self._powell.update_obj(self._sign * float(f0))

    def _add_data(self, new_data: pd.DataFrame):
        # first update the rcds object from the last measurement
        res = float(new_data.iloc[-1][self.vocs.objective_names].to_numpy())

        if self._powell is not None:
            self._powell.update_obj(self._sign * res)

    def _set_data(self, data):
        self.data = data

    def _generate(self, first_gen: bool = False):
        """generate a new candidate"""
        if first_gen or self._powell is None:
            # first generation or no powell object
            self.reset()

        _x_next = self._powell.propose()
        # Verify the candidate here
        while np.any(_x_next > 1) or np.any(_x_next < 0):
            self._powell.update_obj(
                np.nan
            )  # notify RCDS that the search reached the bound
            _x_next = self._powell.propose()  # request next candidate

        # RCDS generator yields normalized x so denormalize it here
        _x_next = np.array(_x_next).flatten()  # convert 2D matrix to 1D array
        lb, ub = self.vocs.bounds
        x_next = (ub - lb) * _x_next + lb

        x_next = [float(ele) for ele in x_next]
        return [dict(zip(self.vocs.variable_names, x_next))]
