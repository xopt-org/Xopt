import warnings

from botorch.acquisition import (
    qUpperConfidenceBound,
    ScalarizedPosteriorTransform,
    UpperConfidenceBound,
)
from pydantic import Field, field_validator

from xopt.generators.bayesian.bayesian_generator import (
    BayesianGenerator,
    formatted_base_docstring,
)
from xopt.generators.bayesian.objectives import CustomXoptObjective
from xopt.generators.bayesian.time_dependent import TimeDependentBayesianGenerator
from xopt.generators.bayesian.utils import set_botorch_weights


class UpperConfidenceBoundGenerator(BayesianGenerator):
    name = "upper_confidence_bound"
    beta: float = Field(2.0, description="Beta parameter for UCB optimization")
    supports_batch_generation: bool = True
    __doc__ = """Bayesian optimization generator using Upper Confidence Bound

Attributes
----------
beta : float, default 2.0
    Beta parameter for UCB optimization, controlling the trade-off between exploration
    and exploitation. Higher values of beta prioritize exploration.

    """ + formatted_base_docstring()

    @field_validator("vocs")
    def validate_vocs_without_constraints(cls, v):
        if v.constraints:
            warnings.warn(
                f"Using {cls.__name__} with constraints may lead to numerical issues if the base acquisition "
                f"function has negative values."
            )
        return v

    @field_validator("log_transform_acquisition_function")
    def validate_log_transform_acquisition_function(cls, v):
        if v:
            raise ValueError(
                "Log transform cannot be applied to potentially negative UCB "
                "acquisition function."
            )

    def _get_acquisition(self, model):
        objective = self._get_objective()
        if self.n_candidates > 1 or isinstance(objective, CustomXoptObjective):
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
            weights = set_botorch_weights(self.vocs)
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
