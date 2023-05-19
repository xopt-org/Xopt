from botorch.acquisition import qUpperConfidenceBound
from pydantic import Field

from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.custom_botorch.constrained_acqusition import (
    ConstrainedMCAcquisitionFunction,
)
from xopt.generators.bayesian.options import AcquisitionOptions
from xopt.generators.bayesian.time_dependent import (
    TimeDependentAcquisitionOptions,
    TimeDependentBayesianGenerator,
)


class UpperConfidenceBoundOptions(AcquisitionOptions):
    beta: float = Field(2.0, description="Beta parameter for UCB optimization")


class TDUpperConfidenceBoundOptions(TimeDependentAcquisitionOptions):
    beta: float = Field(2.0, description="Beta parameter for UCB optimization")


class UpperConfidenceBoundGenerator(BayesianGenerator):
    name = "upper_confidence_bound"
    acquisition_options: UpperConfidenceBoundOptions = UpperConfidenceBoundOptions()
    supports_batch_generation = True
    __doc__ = """Implements Bayeisan Optimization using the Upper Confidence Bound
    acquisition function"""

    def _get_acquisition(self, model):
        sampler = self._get_sampler(model)
        qUCB = qUpperConfidenceBound(
            model,
            sampler=sampler,
            objective=self._get_objective(),
            beta=self.acquisition_options.beta,
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
    acquisition_options: TimeDependentAcquisitionOptions = (
        TimeDependentAcquisitionOptions()
    )
    __doc__ = """Implements Time-Dependent Bayeisan Optimization using the Upper
            Confidence Bound acquisition function"""
