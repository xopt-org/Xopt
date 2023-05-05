# +
import logging

from pydantic import Field

from xopt.generators.bayesian.bax.acquisition import ExpectedInformationGain
from xopt.generators.bayesian.bax.algorithms import Algorithm

from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.options import AcqOptions, BayesianOptions

from xopt.vocs import VOCS

logger = logging.getLogger()


class BayesianAlgorithmExecutionOptions(AcqOptions):
    """Options for defining the acquisition function in BO"""

    algo: Algorithm = Field(
        description="Class with functions for Bayesian execution of some algorithm."
    )

    class Config:
        arbitrary_types_allowed = True


class BaxOptions(BayesianOptions):
    acq: BayesianAlgorithmExecutionOptions


class BaxGenerator(BayesianGenerator):
    alias = "BAX"

    def __init__(
        self,
        vocs: VOCS,
        options: BayesianOptions = None,
    ):
        """
        Generator that uses Expected Information Gain acquisition function
        constructed via Bayesian Algorithm Execution.
        For more information, see:
        https://arxiv.org/pdf/2104.09460.pdf

        Parameters
        ----------
        vocs: dict
            Standard vocs dictionary for xopt

        options: BayesianOptions
            Specific options for this generator
        """

        if vocs.n_objectives != 1:
            raise ValueError("vocs must have one objective for optimization")

        super().__init__(vocs, options)
        self.algo = self.options.acq.algo

    def _get_acquisition(self, model):
        single_task_model = model.models[0]
        eig = ExpectedInformationGain(single_task_model, self.algo)
        self.algo_results = eig.algo_results
        return eig
