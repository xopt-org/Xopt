from botorch.acquisition import qUpperConfidenceBound

from xopt import VOCS
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from .options import AcqOptions, BayesianOptions


class UCBOptions(AcqOptions):
    beta: float = 2.0


class UpperConfidenceBoundGenerator(BayesianGenerator):
    def __init__(self, vocs: VOCS, **kwargs):
        """
        Generator using UpperConfidenceBound acquisition function

        Parameters
        ----------
        vocs: dict
            Standard vocs dictionary for xopt

        beta: float, default: 2.0
            Beta value for UCB acquisition function

        maximize: bool, default: False
            If True attempt to maximize the function

        **kwargs
            Keyword arguments passed to SingleTaskGP model

        """
        options = BayesianOptions(
            acq=UCBOptions()
        )

        super(UpperConfidenceBoundGenerator, self).__init__(vocs, options, **kwargs)

    def _get_acquisition(self, model):
        return qUpperConfidenceBound(model, **self.options.acq.dict())
