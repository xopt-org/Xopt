import logging
from typing import Union, Dict

from pydantic import Field

from xopt.generators.bayesian.bax.acquisition import ExpectedInformationGain
from xopt.generators.bayesian.bax.algorithms import Algorithm
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator

logger = logging.getLogger()


class BaxGenerator(BayesianGenerator):
    alias = "BAX"
    algorithm: Algorithm
    algorithm_results: Dict = None

    def _get_acquisition(self, model):
        single_task_model = model.models[0]
        eig = ExpectedInformationGain(
            single_task_model, self.algorithm, self._get_optimization_bounds()
        )
        self.algorithm_results = eig.algorithm_results
        return eig

