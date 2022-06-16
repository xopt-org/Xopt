from botorch.acquisition import qUpperConfidenceBound
from pydantic import Field

from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.custom_botorch.constrained_acqusition import (
    ConstrainedMCAcquisitionFunction,
)
from xopt.generators.bayesian.objectives import (
    create_constraint_callables,
    create_mc_objective,
)
from xopt.generators.bayesian.options import AcqOptions, BayesianOptions
from xopt.vocs import VOCS


class UpperConfidenceBoundOptions(AcqOptions):
    beta: float = Field(2.0, description="Beta parameter for UCB optimization")


class UCBOptions(BayesianOptions):
    acq = UpperConfidenceBoundOptions()


class UpperConfidenceBoundGenerator(BayesianGenerator):
    alias = "upper_confidence_bound"

    def __init__(self, vocs: VOCS, options: UCBOptions = UCBOptions()):
        """
        Generator using UpperConfidenceBound acquisition function

        Parameters
        ----------
        vocs: dict
            Standard vocs dictionary for xopt

        options: UpperConfidenceBoundOptions
            Specific options for this generator
        """
        if not isinstance(options, UCBOptions):
            raise ValueError("options must be a UCBOptions object")

        if vocs.n_objectives != 1:
            raise ValueError("vocs must have one objective for optimization")

        super(UpperConfidenceBoundGenerator, self).__init__(vocs, options)

    @staticmethod
    def default_options() -> UCBOptions:
        return UCBOptions()

    def _get_objective(self):
        return create_mc_objective(self.vocs)

    def _get_acquisition(self, model):
        qUCB = qUpperConfidenceBound(
            model,
            sampler=self.sampler,
            objective=self.objective,
            beta=self.options.acq.beta,
        )

        cqUCB = ConstrainedMCAcquisitionFunction(
            model, qUCB, create_constraint_callables(self.vocs), infeasible_cost=0.0
        )

        return cqUCB
