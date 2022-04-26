from botorch.acquisition import qUpperConfidenceBound

from xopt.vocs import VOCS
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.options import AcqOptions, BayesianOptions


class UpperConfidenceBoundOptions(AcqOptions):
    beta: float = 2.0


class UCBOptions(BayesianOptions):
    acq = UpperConfidenceBoundOptions()


class UpperConfidenceBoundGenerator(BayesianGenerator):
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
        super(UpperConfidenceBoundGenerator, self).__init__(vocs, options)

    def _get_acquisition(self, model):
        return qUpperConfidenceBound(model, **self.options.acq.dict())
