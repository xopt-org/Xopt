from botorch.acquisition import qUpperConfidenceBound
from pydantic import Field

from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.custom_botorch.constrained_acqusition import (
    ConstrainedMCAcquisitionFunction,
)
from xopt.generators.bayesian.options import AcqOptions, BayesianOptions
from xopt.generators.bayesian.time_dependent import (
    TimeDependentAcqOptions,
    TimeDependentBayesianGenerator,
    TimeDependentModelOptions,
    TimeDependentOptions,
)
from xopt.utils import format_option_descriptions
from xopt.vocs import VOCS


class UpperConfidenceBoundOptions(AcqOptions):
    beta: float = Field(2.0, description="Beta parameter for UCB optimization")


class TDUpperConfidenceBoundOptions(TimeDependentAcqOptions):
    beta: float = Field(2.0, description="Beta parameter for UCB optimization")


class UCBOptions(BayesianOptions):
    acq = UpperConfidenceBoundOptions()


class TDUCBOptions(UCBOptions, TimeDependentOptions):
    acq = TDUpperConfidenceBoundOptions()
    model = TimeDependentModelOptions()


class UpperConfidenceBoundGenerator(BayesianGenerator):
    alias = "upper_confidence_bound"
    __doc__ = (
        """Implements Bayeisan Optimization using the Upper Confidence Bound
    acquisition function"""
        + f"{format_option_descriptions(UCBOptions())}"
    )

    def __init__(self, vocs: VOCS, options: UCBOptions = None):
        """
        Generator using UpperConfidenceBound acquisition function

        Parameters
        ----------
        vocs: dict
            Standard vocs dictionary for xopt

        options: UpperConfidenceBoundOptions
            Specific options for this generator
        """
        options = options or UCBOptions()
        if not isinstance(options, UCBOptions):
            raise ValueError("options must be a UCBOptions object")

        if vocs.n_objectives != 1:
            raise ValueError("vocs must have one objective for optimization")

        super().__init__(vocs, options)

    @staticmethod
    def default_options() -> BayesianOptions:
        return UCBOptions()

    def _get_acquisition(self, model):
        qUCB = qUpperConfidenceBound(
            model,
            sampler=self.sampler,
            objective=self._get_objective(),
            beta=self.options.acq.beta,
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
    alias = "time_dependent_upper_confidence_bound"
    __doc__ = (
        """Implements Time-Dependent Bayeisan Optimization using the Upper
            Confidence Bound acquisition function"""
        + f"{format_option_descriptions(TDUCBOptions())}"
    )

    def __init__(self, vocs: VOCS, options: TDUCBOptions = None):
        options = options or TDUCBOptions()
        if not type(options) is TDUCBOptions:
            raise ValueError("options must be a TDUCBOptions object")

        super(TDUpperConfidenceBoundGenerator, self).__init__(vocs, options)

    @staticmethod
    def default_options() -> TDUCBOptions:
        return TDUCBOptions()
