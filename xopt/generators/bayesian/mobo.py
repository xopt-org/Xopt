from typing import Dict

import torch
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning
from pydantic import Field

from xopt.generators.bayesian.objectives import create_mobo_objective
from xopt.vocs import VOCS
from ...errors import XoptError
from ...utils import format_option_descriptions
from .bayesian_generator import BayesianGenerator
from .options import AcqOptions, BayesianOptions
from .utils import set_botorch_weights


class MOBOAcqOptions(AcqOptions):
    reference_point: Dict[str, float] = Field(
        None, description="dict of reference points for multi-objective optimization"
    )


class MOBOOptions(BayesianOptions):
    acq = MOBOAcqOptions()


class MOBOGenerator(BayesianGenerator):
    alias = "mobo"
    __doc__ = (
        """Implements Multi-Objective Bayesian Optimization using the Expected
            Hypervolume Improvement acquisition function"""
        + f"{format_option_descriptions(MOBOOptions())}"
    )

    def __init__(self, vocs: VOCS, options: MOBOOptions = None):
        options = options or MOBOOptions()
        if not isinstance(options, MOBOOptions):
            raise ValueError("options must be a MOBOOptions object")

        super().__init__(vocs, options, supports_batch_generation=True)

    @staticmethod
    def default_options() -> MOBOOptions:
        return MOBOOptions()

    @property
    def reference_point(self):
        if self.options.acq.reference_point is None:
            raise XoptError(
                "referenece point must be specified for multi-objective " "algorithm"
            )

        pt = []
        for name in self.vocs.objective_names:
            ref_val = self.options.acq.reference_point[name]
            if self.vocs.objectives[name] == "MINIMIZE":
                pt += [-ref_val]
            elif self.vocs.objectives[name] == "MAXIMIZE":
                pt += [ref_val]
            else:
                raise ValueError(
                    f"objective type {self.vocs.objectives[name]} not\
                        supported"
                )

        return torch.tensor(pt, **self._tkwargs)

    def _get_objective(self):
        return create_mobo_objective(self.vocs, self._tkwargs)

    def _get_acquisition(self, model):
        inputs = self.get_input_data(self.data)

        # fix problem with qNEHVI interpretation with constraints
        acq = qNoisyExpectedHypervolumeImprovement(
            model,
            X_baseline=inputs,
            constraints=self._get_constraint_callables(),
            ref_point=self.reference_point,
            sampler=self.sampler,
            objective=self._get_objective(),
            cache_root=False,
            prune_baseline=True,
        )

        return acq

    def calculate_hypervolume(self):
        """compute hypervolume given data"""
        objective_data = torch.tensor(
            self.vocs.objective_data(self.data, return_raw=True).to_numpy()
        )

        # hypervolume must only take into account feasible data points
        if self.vocs.n_constraints > 0:
            objective_data = objective_data[
                self.vocs.feasibility_data(self.data)["feasible"].to_list()
            ]

        n_objectives = self.vocs.n_objectives
        weights = torch.zeros(n_objectives)
        weights = set_botorch_weights(weights, self.vocs)
        objective_data = objective_data * weights

        # compute hypervolume
        bd = DominatedPartitioning(ref_point=self.reference_point, Y=objective_data)
        volume = bd.compute_hypervolume().item()

        return volume
