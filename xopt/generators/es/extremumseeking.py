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
    amplitude: PositiveFloat = Field(1.0, description="dither amplitude")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_data(self, new_data: pd.DataFrame):
        assert (
            len(new_data) == 1
        ), f"length of new_data must be 1, found: {len(new_data)}"

        self.data = new_data.iloc[-1:]

    def _get_diff_ave(self):
        bound_low, bound_up = self.vocs.bounds
        p_ave = (bound_up + bound_low) / 2
        p_diff = bound_up - bound_low
        return p_ave, p_diff

    def _get_last_input_outcome(self):
        last_input = self.data[self.vocs.variable_names].to_numpy()[0]
        last_outcome = self.vocs.objective_data(self.data).to_numpy()[0, 0]
        return last_input, last_outcome

    # Function that normalizes paramters
    def p_normalize(self, p):
        p_ave, p_diff = self._get_diff_ave()
        p_norm = 2.0 * (p - p_ave) / p_diff
        return p_norm

    # Function that un-normalizes parameters
    def p_un_normalize(self, p):
        p_ave, p_diff = self._get_diff_ave()
        p_un_norm = p * p_diff / 2.0 + p_ave
        return p_un_norm

    def generate(self, n_candidates) -> pd.DataFrame:
        if n_candidates != 1:
            raise NotImplementedError(
                "extremum seeking can only produce one candidate at a time"
            )

        p_ave, p_diff = self._get_diff_ave()

        # Initial data point
        if self.data.empty:
            return pd.DataFrame(
                dict(zip(self.vocs.variable_names, p_ave.reshape(-1, 1)))
            )

        last_input, last_outcome = self._get_last_input_outcome()
        step_number = len(self.data) - 1

        p_n = self.p_normalize(last_input)

        nES = len(self.vocs.variables)
        wES = np.linspace(1.0, 1.75, int(np.ceil(nES / 2)))
        dtES = 2 * np.pi / (10 * np.max(wES))
        aES = np.zeros(nES)
        for n in np.arange(nES):
            jw = int(np.floor(n / 2))
            aES[n] = wES[jw] * self.oscillation_size**2

        # ES step for each parameter
        p_next_n = np.zeros(nES)

        # Loop through each parameter
        for j in np.arange(nES):
            # Use the same frequency for each two parameters
            # Alternating Sine and Cosine
            jw = int(np.floor(j / 2))
            if not j % 2:
                p_next_n[j] = p_n[j] + self.amplitude * dtES * np.cos(
                    dtES * step_number * wES[jw] + self.k * last_outcome
                ) * np.sqrt(aES[j] * wES[jw])
            else:
                p_next_n[j] = p_n[j] + self.amplitude * dtES * np.sin(
                    dtES * step_number * wES[jw] + self.k * last_outcome
                ) * np.sqrt(aES[j] * wES[jw])

            # For each new ES value, check that we stay within min/max constraints
            if p_next_n[j] < -1.0:
                p_next_n[j] = -1.0
            if p_next_n[j] > 1.0:
                p_next_n[j] = 1.0

        p_next = self.p_un_normalize(p_next_n)

        self.amplitude *= self.decay_rate  # decay the osc amplitude

        # Return the next value
        return pd.DataFrame(dict(zip(self.vocs.variable_names, p_next.reshape(-1, 1))))
