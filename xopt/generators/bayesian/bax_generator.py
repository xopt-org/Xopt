import logging
import pickle
from copy import deepcopy
from typing import Dict

from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from xopt.generators.bayesian.bax.acquisition import ModelListExpectedInformationGain
from xopt.generators.bayesian.bax.algorithms import Algorithm
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.turbo import EntropyTurboController, SafetyTurboController
from xopt.generators.bayesian.utils import validate_turbo_controller_base

logger = logging.getLogger()


class BaxGenerator(BayesianGenerator):
    name = "BAX"
    algorithm: Algorithm = Field(description="algorithm evaluated in the BAX process")
    algorithm_results: Dict = Field(
        None, description="dictionary results from algorithm", exclude=True
    )
    algorithm_results_file: str = Field(
        None, description="file name to save algorithm results at every step"
    )
    _n_calls: int = 0

    @field_validator("turbo_controller", mode="before")
    def validate_turbo_controller(cls, value, info: ValidationInfo):
        """note default behavior is no use of turbo"""
        controller_dict = {
            "entropy": EntropyTurboController,
            "safety": SafetyTurboController,
        }
        value = validate_turbo_controller_base(value, controller_dict, info)

        return value

    @field_validator("vocs", mode="after")
    def validate_vocs(cls, v, info: ValidationInfo):
        if v.n_objectives != 0:
            raise ValueError("this generator only supports observables")
        return v

    def generate(self, n_candidates: int) -> list[dict]:
        self._n_calls += 1
        return super().generate(n_candidates)

    def _get_acquisition(self, model):
        bax_model_ids = [
            self.vocs.output_names.index(name)
            for name in self.algorithm.observable_names_ordered
        ]
        bax_model = model.subset_output(bax_model_ids)
        eig = ModelListExpectedInformationGain(
            bax_model, self.algorithm, self._get_optimization_bounds()
        )
        self.algorithm_results = eig.algorithm_results
        if self.algorithm_results_file is not None:
            results = deepcopy(self.algorithm_results)

            with open(
                f"{self.algorithm_results_file}_{self._n_calls}.pkl", "wb"
            ) as outfile:
                pickle.dump(results, outfile, protocol=pickle.HIGHEST_PROTOCOL)

        return eig
