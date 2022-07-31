import time
from abc import ABC
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import FixedFeatureAcquisitionFunction
from gpytorch import Module
from pydantic import Field

from xopt.generators.bayesian import BayesianGenerator
from xopt.generators.bayesian.models.standard import create_standard_model
from xopt.generators.bayesian.options import AcqOptions, BayesianOptions
from xopt.vocs import VOCS


class TDAcqOptions(AcqOptions):
    added_time: float = Field(
        0.0,
        description="time added to current time to get target predcition time",
    )


class TDOptions(BayesianOptions):
    acq = TDAcqOptions()


class TimeDependentBayesianGenerator(BayesianGenerator, ABC):
    def __init__(self, vocs: VOCS, options: TDOptions = TDOptions()):
        super(TimeDependentBayesianGenerator, self).__init__(vocs, options)
        self.target_prediction_time = None

    def generate(self, n_candidates: int) -> List[Dict]:
        self.target_prediction_time = time.time() + self.options.acq.added_time
        output = super().generate(n_candidates)

        if time.time() > self.target_prediction_time:
            raise RuntimeWarning(
                "target prediction time is in the past! Increase "
                "added time for accurate results"
            )
        while time.time() < self.target_prediction_time:
            time.sleep(0.001)

        return output

    def train_model(self, data: pd.DataFrame, update_internal=True) -> Module:
        """
        Returns a ModelListGP containing independent models for the objectives and
        constraints

        """
        # drop nans
        valid_data = data[
            self.vocs.variable_names + self.vocs.output_names + ["time"]
        ].dropna()

        # create dataframes for processed data
        variable_data = self.vocs.variable_data(valid_data, "")
        # add time column to variable data
        variable_data = pd.concat([variable_data, valid_data["time"]], axis=1)
        # add bounds for input transformation
        bounds = np.hstack(
            [
                self.vocs.bounds,
                np.array(
                    (
                        valid_data["time"].to_numpy().min(),
                        valid_data["time"].to_numpy().max()
                        + 2 * self.options.acq.added_time,
                    )
                ).reshape(2, 1),
            ]
        )

        objective_data = self.vocs.objective_data(valid_data, "")
        constraint_data = self.vocs.constraint_data(valid_data, "")

        _model = create_standard_model(
            variable_data,
            objective_data,
            constraint_data,
            bounds=bounds,
            tkwargs=self._tkwargs,
            **self.options.model.dict(),
        )

        if update_internal:
            self._model = _model
        return _model

    def get_acquisition(self, model):
        acq = super().get_acquisition(model)

        # identify which column has the `time` attribute
        column = [-1]
        value = torch.tensor(self.target_prediction_time, **self._tkwargs).unsqueeze(0)
        fixed_acq = FixedFeatureAcquisitionFunction(
            acq, self.vocs.n_variables + 1, column, value
        )

        return fixed_acq
