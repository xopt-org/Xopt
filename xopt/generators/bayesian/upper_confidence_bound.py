from botorch.acquisition import qUpperConfidenceBound
from pydantic import Field

from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.custom_botorch.constrained_acqusition import (
    ConstrainedMCAcquisitionFunction,
)
from xopt.generators.bayesian.time_dependent import TimeDependentBayesianGenerator


class UpperConfidenceBoundGenerator(BayesianGenerator):
    name = "upper_confidence_bound"
    beta: float = Field(2.0, description="Beta parameter for UCB optimization")
    supports_batch_generation = True
    __doc__ = """Implements Bayeisan Optimization using the Upper Confidence Bound
    acquisition function"""

    def _get_acquisition(self, model):
        sampler = self._get_sampler(model)
        qUCB = qUpperConfidenceBound(
            model,
            sampler=sampler,
            objective=self._get_objective(),
            beta=self.beta,
        )

        cqUCB = ConstrainedMCAcquisitionFunction(
            model,
            qUCB,
            self._get_constraint_callables(),
        )

        return cqUCB


class TDUpperConfidenceBoundGenerator(
    TimeDependentBayesianGenerator, UpperConfidenceBoundGenerator
):
    name = "time_dependent_upper_confidence_bound"
    __doc__ = """Implements Time-Dependent Bayeisan Optimization using the Upper
            Confidence Bound acquisition function"""
