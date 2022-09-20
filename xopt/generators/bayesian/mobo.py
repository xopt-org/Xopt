from typing import List, Dict

import torch
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
from pydantic import Field

from xopt.generators.bayesian.objectives import (
    create_constraint_callables,
    create_mobo_objective,
)

from xopt.vocs import VOCS
from .bayesian_generator import BayesianGenerator
from .options import AcqOptions, BayesianOptions


class MOBOAcqOptions(AcqOptions):
    reference_point: Dict[str, float] = Field(
        None, description="dict of reference points for multi-objective optimization"
    )


class MOBOOptions(BayesianOptions):
    acq = MOBOAcqOptions()


class MOBOGenerator(BayesianGenerator):
    alias = "mobo"

    def __init__(self, vocs: VOCS, options: MOBOOptions = None):
        options = options or MOBOOptions()
        if not isinstance(options, MOBOOptions):
            raise ValueError("options must be a MOBOOptions object")

        super().__init__(vocs, options)

    @staticmethod
    def default_options() -> MOBOOptions:
        return MOBOOptions()

    @property
    def reference_point(self):
        pt = []
        for name, val in self.vocs.objectives.items():
            ref_val = self.options.acq.reference_point[name]
            if val == "MINIMIZE":
                pt += [-ref_val]
            elif val == "MAXIMIZE":
                pt += [ref_val]

        return torch.tensor(pt, **self._tkwargs)

    def _get_objective(self):
        return create_mobo_objective(self.vocs)

    def _get_acquisition(self, model):
        inputs = self.get_input_data(self.data)
        outcomes = self.get_outcome_data(self.data)

        # get list of constraining functions
        constraint_callables = create_constraint_callables(self.vocs)

        acq = qNoisyExpectedHypervolumeImprovement(
            model,
            X_baseline=inputs,
            prune_baseline=True,
            constraints=constraint_callables,
            ref_point=self.reference_point,
            sampler=self.sampler,
            objective=self.objective,
        )

        return acq
