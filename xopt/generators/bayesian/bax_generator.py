# +
import logging
from typing import Callable, Union

from pydantic import Field

from xopt.generators.bayesian.bax.acquisition import ExpectedInformationGain
from xopt.generators.bayesian.bax.algorithms import GridMinimize

from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.options import AcqOptions, BayesianOptions

from xopt.pydantic import XoptBaseModel
from xopt.vocs import VOCS

logger = logging.getLogger()


class AlgorithmOptions(XoptBaseModel):
    """Options for defining the algorithm in BAX"""

    Algo: Callable = Field(
        description="Class constructor for a specific Bayesian algorithm executor"
    )
    n_samples: int = Field(
        20, description="number of posterior samples on which to execute the algorithm"
    )


class GridMinimizeOptions(AlgorithmOptions):
    Algo: Callable = Field(
        GridMinimize,
        description="Class constructor for a specific Bayesian algorithm executor",
    )
    n_steps_sample_grid: Union[int, list[int]] = Field(
        25, description="number of steps to use per dimension for the sample grid scans"
    )


class BayesianAlgorithmExecutionOptions(AcqOptions):
    """Options for defining the acquisition function in BO"""

    algo: AlgorithmOptions = GridMinimizeOptions()


class BaxOptions(BayesianOptions):
    acq = BayesianAlgorithmExecutionOptions()


class BaxGenerator(BayesianGenerator):
    alias = "BAX"

    def __init__(
        self,
        vocs: VOCS,
        options: BaxOptions = None,
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
        options = options or BaxOptions()
        if not isinstance(options, BaxOptions):
            raise ValueError("options must be a BaxOptions object")

        if vocs.n_objectives != 1:
            raise ValueError("vocs must have one objective for optimization")

        super().__init__(vocs, options)
        self.algo_executor = self.construct_algo_executor()

    @staticmethod
    def default_options() -> BayesianOptions:
        return BaxOptions()

    def _get_acquisition(self, model):
        single_task_model = model.models[0]
        eig = ExpectedInformationGain(single_task_model, self.algo_executor)
        self.algo_results = eig.algo_results
        return eig

    def construct_algo_executor(self):
        algo_options = self.options.acq.algo.dict()
        Algo = algo_options.pop("Algo")
        algo = Algo(domain=self.vocs.bounds.T, **algo_options)
        return algo
