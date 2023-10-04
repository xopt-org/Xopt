import warnings

import torch
from botorch.acquisition import (
    qUpperConfidenceBound,
    ScalarizedPosteriorTransform,
    UpperConfidenceBound,
)
from pydantic import Field, field_validator

from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.time_dependent import TimeDependentBayesianGenerator
from xopt.generators.bayesian.utils import set_botorch_weights


class UpperConfidenceBoundGenerator(BayesianGenerator):
    name = "upper_confidence_bound"
    beta: float = Field(2.0, description="Beta parameter for UCB optimization")
    supports_batch_generation = True
    __doc__ = """Implements Bayesian Optimization using the Upper Confidence Bound
    acquisition function"""

    @field_validator("vocs")
    def validate_vocs_without_constraints(cls, v):
        if v.constraints:
            warnings.warn(
                f"Using {cls.__name__} with constraints may lead to numerical issues if the base acquisition "
                f"function has negative values."
            )
        return v

    def _get_acquisition(self, model):
        if self.n_candidates > 1:
            # MC sampling for generating multiple candidate points
            sampler = self._get_sampler(model)
            acq = qUpperConfidenceBound(
                model,
                sampler=sampler,
                objective=self._get_objective(),
                beta=self.beta,
            )
        else:
            # analytic acquisition function for single candidate generation
            weights = torch.zeros(self.vocs.n_outputs).to(**self._tkwargs)
            weights = set_botorch_weights(weights, self.vocs)
            posterior_transform = ScalarizedPosteriorTransform(weights)
            acq = UpperConfidenceBound(
                model, beta=self.beta, posterior_transform=posterior_transform
            )

        return acq


class TDUpperConfidenceBoundGenerator(
    TimeDependentBayesianGenerator, UpperConfidenceBoundGenerator
):
    name = "time_dependent_upper_confidence_bound"
    __doc__ = """Implements Time-Dependent Bayesian Optimization using the Upper
            Confidence Bound acquisition function"""
