import time
import warnings
from abc import ABC

import pandas as pd
import torch
from botorch.acquisition import FixedFeatureAcquisitionFunction
from pydantic import Field, PositiveFloat

from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.options import AcquisitionOptions


class TimeDependentAcquisitionOptions(AcquisitionOptions):
    added_time: float = Field(
        0.0,
        description="time added to current time to get target predcition time",
    )


class TimeDependentBayesianGenerator(BayesianGenerator, ABC):
    name = "time_dependent_bayesian_generator"
    acquisition_options: TimeDependentAcquisitionOptions = (
        TimeDependentAcquisitionOptions()
    )
    target_prediction_time: PositiveFloat = Field(None)

    def get_input_data(self, data: pd.DataFrame):
        return torch.tensor(
            data[self.vocs.variable_names + ["time"]].to_numpy(), **self._tkwargs
        )

    def generate(self, n_candidates: int) -> pd.DataFrame:
        self.target_prediction_time = time.time() + self.acquisition_options.added_time
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
