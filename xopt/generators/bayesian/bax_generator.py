import importlib
import logging
import pickle
from copy import deepcopy
from typing import Any, Hashable, Optional, cast

from botorch.models import ModelListGP, SingleTaskGP
from gpytorch import Module
from pydantic import (
    Field,
    SerializeAsAny,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic.fields import ModelPrivateAttr, PrivateAttr
from xopt.errors import VOCSError
from xopt.generators.bayesian.bax.acquisition import ModelListExpectedInformationGain
from xopt.generators.bayesian.bax.algorithms import Algorithm, GridOptimize
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.turbo import (
    EntropyTurboController,
    SafetyTurboController,
    TurboController,
)
from xopt.generators.bayesian.utils import validate_turbo_controller_center
from xopt.vocs import VOCS

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
        default=GridOptimize(), description="algorithm evaluated in the BAX process"
    )
    algorithm_results: Optional[dict] = Field(
        None, description="dictionary results from algorithm", exclude=True
    )
    algorithm_results_file: Optional[str] = Field(
        None, description="file name to save algorithm results at every step"
    )
    _n_calls: int = 0
    _compatible_turbo_controllers: list[type[TurboController]] = PrivateAttr(
        default=[EntropyTurboController, SafetyTurboController]
    )

    # NOTE: this is meant for use in Badger, TODO: add it to Xopt
    _compatible_algorithms: list[type[Algorithm]] = PrivateAttr(default=[GridOptimize])

    @field_validator("vocs", mode="after")
    @classmethod
    def validate_vocs(cls, v: VOCS, info: ValidationInfo) -> VOCS:
        # Preserve inherited Bayesian VOCS validation behavior.
        # v = super().validate_vocs(v, info)

        # assert that the generator had no objectives
        if not v.n_objectives == 0:
            raise VOCSError("BAX generator only supports problems with no objectives")

        return v

    @field_validator("algorithm", mode="before")
    @classmethod
    def validate_algorithm(cls, v: Any, info: ValidationInfo) -> Any:
        if isinstance(v, dict):
            if "class_path" in v:
                class_path = v.pop("class_path")
                module_name, class_name = class_path.rsplit(".", 1)
                try:
                    algorithm_class = getattr(
                        importlib.import_module(module_name), class_name
                    )
                except ModuleNotFoundError:
                    raise ValueError(f"Cannot import '{module_name}.{class_name}'")
            elif "name" in v:
                name = v["name"]
                algorithm_class = next(
                    (
                        c
                        for c in cls._compatible_algorithms.default
                        if c.model_fields["name"].default == name
                    ),
                    None,
                )
                if algorithm_class is None:
                    raise ValueError(
                        f"Unknown algorithm name '{name}'. "
                        f"Provide one of {[c.model_fields['name'].default for c in cls._compatible_algorithms.default]} "
                        f"or supply 'class_path'."
                    )
            else:
                raise ValueError(
                    "Algorithm dictionary must contain 'class_path' or 'name' key"
                )

            v = algorithm_class.model_validate(v)

        return v

    @model_validator(mode="after")
    def validate_model_after(self) -> "BaxGenerator":
        # validate turbo controller center if it exists
        validate_turbo_controller_center(self)

        return self

    @classmethod
    def get_compatible_algorithms(cls) -> list[type[Algorithm]]:
        compatible = cls._compatible_algorithms
        compatible_list: list[type[Algorithm]] = []
        if isinstance(compatible, ModelPrivateAttr):
            compatible_list = cast(list[type[Algorithm]], compatible.get_default())
        return compatible_list

    def generate(self, n_candidates: int) -> list[dict[Hashable, Any]]:
        """
        Generate a specified number of candidate samples.

        Parameters
        ----------
        n_candidates : int
            The number of candidate samples to generate.

        Returns
        -------
        list[dict[Hashable, Any]]
            A list of dictionaries containing the generated samples.
        """
        self._n_calls += 1
        return super().generate(n_candidates)

    def _get_acquisition(self, model: Module) -> ModelListExpectedInformationGain:
        """
        Get the acquisition function.

        Parameters
        ----------
        model : Module
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
