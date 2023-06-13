from abc import ABC
from typing import Dict, List

import pandas as pd
import torch
from botorch.acquisition import FixedFeatureAcquisitionFunction

from xopt.errors import XoptError
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.models.contextual_model import ContextualModelOptions
from xopt.generators.bayesian.options import BayesianOptions
from xopt.generators.bayesian.upper_confidence_bound import (
    UCBOptions,
    UpperConfidenceBoundGenerator,
)
from xopt.utils import format_option_descriptions
from xopt.vocs import VOCS


class ContextualOptions(BayesianOptions):
    model = ContextualModelOptions()


class ContextualBayesianGenerator(BayesianGenerator, ABC):
    def __init__(self, vocs: VOCS, options: BayesianOptions = None):
        options = options or BayesianOptions()
        if not isinstance(options, BayesianOptions):
            raise ValueError("options must be a TDOptions object")

        super().__init__(vocs, options)
        self.target_prediction_time = None

    def get_input_data(self, data: pd.DataFrame):
        return torch.tensor(
            data[self.vocs.variable_names + ["context"]].to_numpy(), **self._tkwargs
        )

    def generate(self, n_candidates: int) -> List[Dict]:
        output = super().generate(n_candidates)
        return output

    def get_acquisition(self, model):
        acq = super().get_acquisition(model)
        # use the latest context assuming that context varies slowly
        latest_context = self.data[["context"]].to_numpy()[-1]
        # get context variables
        len_context = len(latest_context)
        column = list(range(-len_context, 0))
        value = torch.tensor(latest_context, **self._tkwargs).unsqueeze(0)
        fixed_acq = FixedFeatureAcquisitionFunction(
            acq, self.vocs.n_variables + len_context, column, value
        )
        return fixed_acq

    def _get_initial_batch_points(self, bounds):
        if self.options.optim.use_nearby_initial_points:
            raise XoptError(
                "nearby initial points not implemented for " "contextual optimization"
            )
        else:
            batch_initial_points = None
            raw_samples = self.options.optim.raw_samples
        return batch_initial_points, raw_samples


class ContextualUCBOptions(UCBOptions, ContextualOptions):
    model = ContextualModelOptions()


class ContextualUpperConfidenceBoundGenerator(
    ContextualBayesianGenerator, UpperConfidenceBoundGenerator
):
    alias = "time_dependent_upper_confidence_bound"
    __doc__ = (
        """Implements Time-Dependent Bayeisan Optimization using the Upper
            Confidence Bound acquisition function"""
        + f"{format_option_descriptions(UCBOptions())}"
    )

    def __init__(self, vocs: VOCS, options: ContextualUCBOptions = None):
        options = options or ContextualUCBOptions()
        if not type(options) is ContextualUCBOptions:
            raise ValueError("options must be a ContextualUCBOptions object")

        super(UpperConfidenceBoundGenerator, self).__init__(vocs, options)
