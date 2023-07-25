import json
import logging
import pickle
from copy import deepcopy
from typing import Dict

import pandas as pd

from xopt.generators.bayesian.bax.acquisition import ExpectedInformationGain
from xopt.generators.bayesian.bax.algorithms import Algorithm
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator

logger = logging.getLogger()


class BaxGenerator(BayesianGenerator):
    alias = "BAX"
    algorithm: Algorithm
    algorithm_results: Dict = None
    algorithm_results_file: str = None

    _n_calls: int = 0

    class Config:
        underscore_attrs_are_private = True

    def generate(self, n_candidates: int) -> pd.DataFrame:
        self._n_calls += 1
        return super().generate(n_candidates)

    def _get_acquisition(self, model):
        single_task_model = model.models[0]
        eig = ExpectedInformationGain(
            single_task_model, self.algorithm, self._get_optimization_bounds()
        )
        self.algorithm_results = eig.algorithm_results
        if self.algorithm_results_file is not None:
            results = deepcopy(self.algorithm_results)

            with open(
                    f"{self.algorithm_results_file}_{self._n_calls}.pkl", "wb"
            ) as outfile:
                pickle.dump(results, outfile, protocol=pickle.HIGHEST_PROTOCOL)

        return eig
