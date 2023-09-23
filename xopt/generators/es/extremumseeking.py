import logging

import numpy as np
import pandas as pd
from pydantic import Field, PositiveFloat

from xopt.generator import Generator

logger = logging.getLogger(__name__)


class ExtremumSeekingGenerator(Generator):
    """
    Extremum seeking algorithm.

    Reference:
    Extremum Seeking-Based Control System for Particle Accelerator
    Beam Loss Minimization
    A. Scheinker, E. -C. Huang and C. Taylor
    doi: 10.1109/TCST.2021.3136133

    This algorithm must be stepped serially.
    """

    name = "extremum_seeking"
    k: PositiveFloat = Field(2.0, description="feedback gain")
    oscillation_size: PositiveFloat = Field(0.1, description="oscillation size")
    decay_rate: PositiveFloat = Field(1.0, description="decay rate")

    _nES = 0
    _wES = []
    _dtES = 0
    _aES = []
    _p_ave = 0
    _p_diff = 0
    _amplitude = 1
    # Evaluation counter, note that the first point counts as zero in ES
    _i = -1
    # Track the last variables and objectives
    _last_input: np.array = None  # 1d numpy array
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

    def add_data(self, new_data: pd.DataFrame):
        assert (
            len(new_data) <= 1
        ), f"length of new_data must be 1, found: {len(new_data)}"

        self.data = new_data.iloc[-1:]
        self._last_input = self.data[self.vocs.variable_names].to_numpy()[0]

        res = self.vocs.objective_data(new_data).to_numpy()
        assert res.shape == (1, 1)
        self._last_outcome = res[0, 0]

        self._i += 1

    # Function that normalizes paramters
    def p_normalize(self, p):
        p_norm = 2.0 * (p - self._p_ave) / self._p_diff
        return p_norm

    # Function that un-normalizes parameters
    def p_un_normalize(self, p):
        p_un_norm = p * self._p_diff / 2.0 + self._p_ave
        return p_un_norm

    def generate(self, n_candidates) -> list[dict]:
        if n_candidates != 1:
            raise NotImplementedError(
                "extremum seeking can only produce one candidate at a time"
            )

        # Initial data point
        if self.data is None:
            return [dict(zip(self.vocs.variable_names, self._p_ave.reshape(-1, 1)))]

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
        return [dict(zip(self.vocs.variable_names, p_next.reshape(-1, 1)))]
