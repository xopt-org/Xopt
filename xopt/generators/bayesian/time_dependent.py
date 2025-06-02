import time
import warnings
from abc import ABC
from copy import copy
from typing import Optional, List

import pandas as pd
import torch
from botorch.acquisition import FixedFeatureAcquisitionFunction
from pydantic import Field, ValidationInfo, field_validator, PositiveFloat

from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.models.time_dependent import TimeDependentModelConstructor


class TimeDependentBayesianGenerator(BayesianGenerator, ABC):
    """
    Time-dependent Bayesian generator for Bayesian Optimization.

    Attributes
    ----------
    name : str
        The name of the generator.
    target_prediction_time : Optional[PositiveFloat]
        The target prediction time.
    added_time : PositiveFloat
        Time added to the current time to get the target prediction time.
    gp_constructor : TimeDependentModelConstructor
        Constructor used to generate the model.
    forgetting_time : Optional[PositiveFloat]
        Time period to forget historical data in seconds.

    Methods
    -------
    validate_gp_constructor(cls, value)
        Validate the Gaussian Process (GP) constructor.
    get_training_data(self, data: pd.DataFrame) -> pd.DataFrame
        Window data based on the forgetting time.
    generate(self, n_candidates: int) -> List[dict]
        Generate candidates for Bayesian Optimization.
    get_input_data(self, data: pd.DataFrame) -> torch.Tensor
        Convert input data to a torch tensor.
    get_acquisition(self, model) -> FixedFeatureAcquisitionFunction
        Get the acquisition function for Bayesian Optimization.
    """

    name = "time_dependent_bayesian_generator"
    target_prediction_time: Optional[PositiveFloat] = Field(None)
    added_time: PositiveFloat = Field(
        0.1,
        description="time added to current time to get target prediction time",
    )

    gp_constructor: TimeDependentModelConstructor = Field(
        TimeDependentModelConstructor(),
        description="constructor used to generate model",
    )
    forgetting_time: Optional[PositiveFloat] = Field(
        None, description="time period to forget historical data in seconds"
    )

    @field_validator("vocs", mode="after")
    def validate_vocs(cls, v, info: ValidationInfo):
        if v.n_objectives != 1:
            raise ValueError("this generator only supports vocs with 1 objective")
        return v

    @field_validator("gp_constructor", mode="before")
    def validate_gp_constructor(
        cls, value: Optional[TimeDependentModelConstructor]
    ) -> TimeDependentModelConstructor:
        """
        Validate the Gaussian Process (GP) constructor.

        Parameters
        ----------
        value : Optional[TimeDependentModelConstructor]
            The GP constructor to validate.

        Returns
        -------
        TimeDependentModelConstructor
            The validated GP constructor.

        Raises
        ------
        ValueError
            If the GP constructor is not found.
        """
        constructor_dict = {"time_dependent": TimeDependentModelConstructor}
        if value is None:
            value = TimeDependentModelConstructor()
        elif isinstance(value, TimeDependentModelConstructor):
            value = value
        elif isinstance(value, str):
            if value in constructor_dict:
                value = constructor_dict[value]()
            else:
                raise ValueError(f"{value} not found")
        elif isinstance(value, dict):
            name = value.pop("name")
            if name in constructor_dict:
                value = constructor_dict[name](**value)
            else:
                raise ValueError(f"{value} not found")

        return value

    def get_training_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Window data based on the forgetting time.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.

        Returns
        -------
        pd.DataFrame
            The windowed data.
        """
        new_data = copy(data)
        if self.forgetting_time is not None:
            new_data = new_data[data["time"] > time.time() - self.forgetting_time]

        return new_data

    def generate(self, n_candidates: int) -> List[dict]:
        """
        Generate candidates for Bayesian Optimization.

        Parameters
        ----------
        n_candidates : int
            The number of candidates to generate.

        Returns
        -------
        List[dict]
            The generated candidates.
        """
        self.target_prediction_time = time.time() + self.added_time
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

    def get_input_data(self, data: pd.DataFrame) -> torch.Tensor:
        """
        Convert input data to a torch tensor.

        Parameters
        ----------
        data : pd.DataFrame
            The input data in the form of a pandas DataFrame.

        Returns
        -------
        torch.Tensor
            A torch tensor containing the input data.

        Notes
        -----
        This method takes a pandas DataFrame as input data and converts it into a
        torch tensor. It specifically selects columns corresponding to the model's
        input names (variables), and the resulting tensor is configured with the data
        type and device settings from the generator.
        """
        return torch.tensor(
            data[self.model_input_names + ["time"]].to_numpy(), **self.tkwargs
        )

    def get_acquisition(
        self, model: torch.nn.Module
    ) -> FixedFeatureAcquisitionFunction:
        """
        Get the acquisition function for Bayesian Optimization.

        Parameters
        ----------
        model : torch.nn.Module
            The model used for Bayesian Optimization.

        Returns
        -------
        FixedFeatureAcquisitionFunction
            The acquisition function with fixed features.
        """
        acq = super().get_acquisition(model)

        # identify which column has the `time` attribute
        column = [-1]
        value = torch.tensor(self.target_prediction_time, **self.tkwargs).unsqueeze(0)
        fixed_acq = FixedFeatureAcquisitionFunction(
            acq, self.vocs.n_variables + 1, column, value
        )

        return fixed_acq
