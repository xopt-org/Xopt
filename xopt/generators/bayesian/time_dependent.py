import time
from abc import ABC
from typing import Callable, Dict, List

import pandas as pd
import torch
from botorch.acquisition import FixedFeatureAcquisitionFunction
from gpytorch import Module
from pydantic import BaseModel, create_model, Field, root_validator

from xopt.generators.bayesian import BayesianGenerator
from xopt.generators.bayesian.models.time_dependent import create_time_dependent_model
from xopt.generators.bayesian.options import AcqOptions, BayesianOptions, ModelOptions
from xopt.utils import get_function, get_function_defaults
from xopt.vocs import VOCS


class TDAcqOptions(AcqOptions):
    added_time: float = Field(
        0.0,
        description="time added to current time to get target predcition time",
    )


class TDModelOptions(ModelOptions):
    """Options for defining the GP model in BO"""

    function: Callable
    kwargs: BaseModel

    @root_validator(pre=True)
    def validate_all(cls, values):
        if "function" in values.keys():
            f = get_function(values["function"])
        else:
            f = create_time_dependent_model

        kwargs = values.get("kwargs", {})
        kwargs = {**get_function_defaults(f), **kwargs}
        # remove add time
        kwargs.pop("added_time")

        values["function"] = f
        values["kwargs"] = create_model("kwargs", **kwargs)()

        return values


class TDOptions(BayesianOptions):
    acq = TDAcqOptions()
    model = TDModelOptions()


class TimeDependentBayesianGenerator(BayesianGenerator, ABC):
    def __init__(self, vocs: VOCS, options: TDOptions = None):
        options = options or TDOptions()
        if not isinstance(options, BayesianOptions):
            raise ValueError("options must be a TDOptions object")

        super().__init__(vocs, options)
        self.target_prediction_time = None

    @staticmethod
    def default_options() -> TDOptions:
        return TDOptions()

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

    def train_model(self, data: pd.DataFrame = None, update_internal=True) -> Module:
        """
        Returns a ModelListGP containing independent models for the objectives and
        constraints

        """
        if data is None:
            data = self.data

        # drop nans
        valid_data = data[
            self.vocs.variable_names + self.vocs.output_names + ["time"]
        ].dropna()

        kwargs = self.options.model.kwargs.dict()

        _model = self.options.model.function(
            valid_data, self.vocs, added_time=self.options.acq.added_time, **kwargs
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
