from typing import List, Dict

import numpy as np
import pandas as pd
from pydantic import Field, PositiveFloat

from xopt.generators.sequential.sequential_generator import SequentialGenerator


class ExtremumSeekingGenerator(SequentialGenerator):
    """
    Extremum seeking algorithm.

    Reference:
    Extremum Seeking-Based Control System for Particle Accelerator
    Beam Loss Minimization
    A. Scheinker, E. -C. Huang and C. Taylor
    doi: 10.1109/TCST.2021.3136133

    This algorithm must be stepped serially.

    Attributes
    ----------
    name : str
        The name of the generator.
    k : PositiveFloat
        Feedback gain.
    oscillation_size : PositiveFloat
        Oscillation size.
    decay_rate : PositiveFloat
        Decay rate.
    _nES : int
        Number of extremum seeking parameters.
    _wES : np.ndarray
        Frequencies for extremum seeking.
    _dtES : float
        Time step for extremum seeking.
    _aES : np.ndarray
        Amplitudes for extremum seeking.
    _p_ave : np.ndarray
        Average of parameter bounds.
    _p_diff : np.ndarray
        Difference of parameter bounds.
    _amplitude : float
        Amplitude of oscillation.
    _i : int
        Evaluation counter.
    _last_input : np.ndarray
        Last input values.
    _last_outcome : float
        Last outcome value.

    Methods
    -------
    add_data(self, new_data: pd.DataFrame)
        Add new data to the generator.
    p_normalize(self, p: np.ndarray) -> np.ndarray
        Normalize parameters.
    p_un_normalize(self, p: np.ndarray) -> np.ndarray
        Un-normalize parameters.
    generate(self, n_candidates: int) -> List[Dict[str, float]]
        Generate a specified number of candidate samples.
    """

    name = "extremum_seeking"
    k: PositiveFloat = Field(2.0, description="feedback gain")
    oscillation_size: PositiveFloat = Field(0.1, description="oscillation size")
    decay_rate: PositiveFloat = Field(1.0, description="decay rate")
    supports_single_objective: bool = True

    _nES: int = 0
    _wES: np.ndarray = np.array([])
    _dtES: float = 0.0
    _aES: np.ndarray = np.array([])
    _p_ave: np.ndarray = np.array([])
    _p_diff: np.ndarray = np.array([])
    _amplitude: float = 1.0
    _i: int = -1
    _last_input: np.ndarray = None
    _last_outcome: float = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._nES = len(self.vocs.variables)
        self._wES = np.linspace(1.0, 1.75, int(np.ceil(self._nES / 2)))
        self._dtES = 2 * np.pi / (10 * np.max(self._wES))
        self._aES = np.zeros(self._nES)
        for n in np.arange(self._nES):
            jw = int(np.floor(n / 2))
            self._aES[n] = self._wES[jw] * (self.oscillation_size) ** 2
        bound_low, bound_up = self.vocs.bounds
        self._p_ave = (bound_up + bound_low) / 2
        self._p_diff = bound_up - bound_low

    def _reset(self):
        self._i = 0
        self._last_input = self.vocs.variable_data(self.data).to_numpy()[-1]
        self._last_outcome = self.vocs.objective_data(self.data).to_numpy()[-1, 0]

    def _add_data(self, new_data: pd.DataFrame):
        self.data = new_data.iloc[-1:]
        self._last_input = self.data[self.vocs.variable_names].to_numpy()[0]

        res = self.vocs.objective_data(new_data).to_numpy()
        self._last_outcome = res[0, 0]

        self._i += 1

    def _set_data(self, data: pd.DataFrame):
        self.data = data.iloc[-1:]

        self._i = 0
        self._last_input = self.vocs.variable_data(self.data).to_numpy()[-1]
        self._last_outcome = self.vocs.objective_data(self.data).to_numpy()[-1, 0]
        # TODO: add proper reload tests

    def p_normalize(self, p: np.ndarray) -> np.ndarray:
        """
        Normalize parameters.

        Parameters
        ----------
        p : np.ndarray
            The parameters to normalize.

        Returns
        -------
        np.ndarray
            The normalized parameters.
        """
        p_norm = 2.0 * (p - self._p_ave) / self._p_diff
        return p_norm

    def p_un_normalize(self, p: np.ndarray) -> np.ndarray:
        """
        Un-normalize parameters.

        Parameters
        ----------
        p : np.ndarray
            The parameters to un-normalize.

        Returns
        -------
        np.ndarray
            The un-normalized parameters.
        """
        p_un_norm = p * self._p_diff / 2.0 + self._p_ave
        return p_un_norm

    def _generate(self, first_gen: bool = False) -> List[Dict[str, float]]:
        """
        Generate next candidate.

        Returns
        -------
        List[Dict[str, float]]
            A list of dictionaries containing the generated samples.
        """
        if first_gen:
            self.reset()

        p_n = self.p_normalize(self._last_input)

        # ES step for each parameter
        p_next_n = np.zeros(self._nES)

        # Loop through each parameter
        for j in np.arange(self._nES):
            # Use the same frequency for each two parameters
            # Alternating Sine and Cosine
            jw = int(np.floor(j / 2))
            if not j % 2:
                p_next_n[j] = p_n[j] + self._amplitude * self._dtES * np.cos(
                    self._dtES * self._i * self._wES[jw] + self.k * self._last_outcome
                ) * np.sqrt(self._aES[j] * self._wES[jw])
            else:
                p_next_n[j] = p_n[j] + self._amplitude * self._dtES * np.sin(
                    self._dtES * self._i * self._wES[jw] + self.k * self._last_outcome
                ) * np.sqrt(self._aES[j] * self._wES[jw])

            # For each new ES value, check that we stay within min/max constraints
            if p_next_n[j] < -1.0:
                p_next_n[j] = -1.0
            if p_next_n[j] > 1.0:
                p_next_n[j] = 1.0

        p_next = self.p_un_normalize(p_next_n)

        self._amplitude *= self.decay_rate  # decay the osc amplitude

        # Return the next value
        p_next = [float(ele) for ele in p_next]
        return [dict(zip(self.vocs.variable_names, p_next))]
