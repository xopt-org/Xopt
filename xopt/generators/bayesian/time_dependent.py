import time
import warnings
from abc import ABC
from typing import Dict, List

import pandas as pd
import torch
from botorch.acquisition import FixedFeatureAcquisitionFunction
from pydantic import Field

from xopt.errors import XoptError
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.models.time_dependent import TimeDependentModelOptions
from xopt.generators.bayesian.options import AcqOptions, BayesianOptions
from xopt.vocs import VOCS


class TimeDependentAcqOptions(AcqOptions):
    added_time: float = Field(
        0.0,
        description="time added to current time to get target predcition time",
    )


class TimeDependentOptions(BayesianOptions):
    acq = TimeDependentAcqOptions()
    model = TimeDependentModelOptions()


class TimeDependentBayesianGenerator(BayesianGenerator, ABC):
    def __init__(self, vocs: VOCS, options: TimeDependentOptions = None):
        options = options or TimeDependentOptions()
        if not isinstance(options, TimeDependentOptions):
            raise ValueError("options must be a TDOptions object")

        super().__init__(vocs, options)
        self.target_prediction_time = None

    @staticmethod
    def default_options() -> TimeDependentOptions:
        return TimeDependentOptions()

    def get_input_data(self, data: pd.DataFrame):
        return torch.tensor(
            data[self.vocs.variable_names + ["time"]].to_numpy(), **self._tkwargs
        )

    def generate(self, n_candidates: int) -> List[Dict]:
        self.target_prediction_time = time.time() + self.options.acq.added_time
        output = super().generate(n_candidates)

        if time.time() > self.target_prediction_time:
            warnings.warn(
                "target prediction time is in the past! Increase "
                "added time for accurate results",
                RuntimeWarning,
            )
        while time.time() < self.target_prediction_time:
            time.sleep(0.001)

        return output

    def get_acquisition(self, model):
        acq = super().get_acquisition(model)

        # identify which column has the `time` attribute
        column = [-1]
        value = torch.tensor(self.target_prediction_time, **self._tkwargs).unsqueeze(0)
        fixed_acq = FixedFeatureAcquisitionFunction(
            acq, self.vocs.n_variables + 1, column, value
        )

        return fixed_acq

    def _get_initial_batch_points(self, bounds):
        if self.options.optim.use_nearby_initial_points:
            raise XoptError(
                "nearby initial points not implemented for "
                "time dependent optimization"
            )
        else:
            batch_initial_points = None
            raw_samples = self.options.optim.raw_samples
        return batch_initial_points, raw_samples
