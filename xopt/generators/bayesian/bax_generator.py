from copy import deepcopy
import importlib
import logging
import pickle
from typing import Dict, List, Optional

from botorch.models import ModelListGP, SingleTaskGP
from pydantic import (
    Field,
    SerializeAsAny,
    ValidationInfo,
    field_validator,
    model_validator,
)

from xopt.errors import VOCSError
from xopt.generators.bayesian.bax.acquisition import ModelListExpectedInformationGain
from xopt.generators.bayesian.bax.algorithms import Algorithm, GridOptimize
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.turbo import EntropyTurboController, SafetyTurboController
from xopt.generators.bayesian.utils import validate_turbo_controller_center

logger = logging.getLogger()


class BaxGenerator(BayesianGenerator):
    """
    BAX Generator for Bayesian optimization.

    Attributes
    ----------
    name : str
        The name of the generator.
    algorithm : Algorithm
        Algorithm evaluated in the BAX process.
    algorithm_results : Dict
        Dictionary results from the algorithm.
    algorithm_results_file : str
        File name to save algorithm results at every step.
    _n_calls : int
        Internal counter for the number of calls to the generate method.

    Methods
    -------
    validate_turbo_controller(cls, value, info: ValidationInfo) -> Any
        Validate the turbo controller.
    validate_vocs(cls, v, info: ValidationInfo) -> VOCS
        Validate the VOCS object.
    generate(self, n_candidates: int) -> List[Dict]
        Generate a specified number of candidate samples.
    _get_acquisition(self, model) -> ModelListExpectedInformationGain
        Get the acquisition function.
    """

    name = "bax"
    supports_constraints: bool = True
    supports_discrete_variables: bool = False
    algorithm: SerializeAsAny[Algorithm] = Field(
        description="algorithm evaluated in the BAX process"
    )
    algorithm_results: Optional[Dict] = Field(
        None, description="dictionary results from algorithm", exclude=True
    )
    algorithm_results_file: Optional[str] = Field(
        None, description="file name to save algorithm results at every step"
    )
    _n_calls: int = 0
    _compatible_turbo_controllers = [EntropyTurboController, SafetyTurboController]

    # NOTE: this is meant for use in Badger, TODO: add it to Xopt
    _compatible_algorithms = [GridOptimize]

    @field_validator("vocs", mode="after")
    def validate_vocs(cls, v, info: ValidationInfo):
        # Preserve inherited Bayesian VOCS validation behavior.
        v = super().validate_vocs(v, info)

        # assert that the generator had no objectives
        if not v.n_objectives == 0:
            raise VOCSError("BAX generator only supports problems with no objectives")

        return v

    @model_validator(mode="after")
    def validate_model_after(self):
        # validate turbo controller center if it exists
        validate_turbo_controller_center(self)

        return self

    @field_validator("algorithm", mode="before")
    def validate_algorithm(cls, v, info: ValidationInfo):
        if isinstance(v, dict):
            try:
                class_path = v.pop("class_path")
                module_name, class_name = class_path.rsplit(".", 1)
            except KeyError:
                raise ValueError("Algorithm dictionary must contain 'class_path' key")

            try:
                algorithm_class = getattr(
                    importlib.import_module(module_name), class_name
                )
            except ModuleNotFoundError:
                raise ValueError(f"Cannot import '{module_name}.{class_name}'")

            v = algorithm_class.model_validate(v)

        return v

    def generate(self, n_candidates: int) -> List[Dict]:
        """
        Generate a specified number of candidate samples.

        Parameters
        ----------
        n_candidates : int
            The number of candidate samples to generate.

        Returns
        -------
        List[Dict]
            A list of dictionaries containing the generated samples.
        """
        self._n_calls += 1
        return super().generate(n_candidates)

    def _get_acquisition(self, model) -> ModelListExpectedInformationGain:
        """
        Get the acquisition function.

        Parameters
        ----------
        model : Model
            The model to use for the acquisition function.

        Returns
        -------
        ModelListExpectedInformationGain
            The acquisition function.
        """
        bax_model_ids = [
            self.vocs.output_names.index(name)
            for name in self.algorithm.observable_names_ordered
        ]
        bax_model = model.subset_output(bax_model_ids)

        if isinstance(bax_model, SingleTaskGP):
            bax_model = ModelListGP(bax_model)

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
