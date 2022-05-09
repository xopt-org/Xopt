from typing import Type

from botorch.acquisition import qUpperConfidenceBound

from xopt.generator import GeneratorOptions
from xopt.vocs import VOCS
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.options import AcqOptions, BayesianOptions
from xopt.generators.bayesian.objectives import create_constrained_mc_objective
from pydantic import Field


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
        return create_constrained_mc_objective(self.vocs)

    def _get_acquisition(self, model):
        return qUpperConfidenceBound(
            model,
            sampler=self.sampler,
            objective=self.objective,
            beta=self.options.acq.beta,
        )
