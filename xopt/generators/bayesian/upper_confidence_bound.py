from botorch.acquisition import qUpperConfidenceBound
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt import VOCS


class UpperConfidenceBoundGenerator(BayesianGenerator):
    def __init__(self, vocs: VOCS, beta=2.0, **kwargs):
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

        super(UpperConfidenceBoundGenerator, self).__init__(
            vocs,
            model_kw=kwargs,
            acqf_kw={'beta': beta}
        )

    def get_acquisition(self, model):
        return qUpperConfidenceBound(
            model,
            **self.acqf_kw
        )
