from typing import List

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
    ref_point: List[float] = Field(
        None, description="reference point for multi-objective optimization"
    )
    use_data_as_reference: bool = Field(
        True,
        description="flag to determine if the dataset determines the reference point",
    )


class MOBOOptions(BayesianOptions):
    acq = MOBOAcqOptions()


class MOBOGenerator(BayesianGenerator):
    alias = "mobo"

    def __init__(self, vocs: VOCS, options: MOBOOptions = MOBOOptions()):
        if not isinstance(options, MOBOOptions):
            raise ValueError("options must be a MOBOOptions object")

        super(MOBOGenerator, self).__init__(vocs, options)

    @staticmethod
    def default_options() -> MOBOOptions:
        return MOBOOptions()

    def _get_objective(self):
        return create_mobo_objective(self.vocs)

    def _get_acquisition(self, model):
        # get reference point from data
        inputs = self.get_input_data(self.data)
        outcomes = self.get_outcome_data(self.data)

        if self.options.acq.use_data_as_reference:
            weighted_points = self.objective(outcomes)
            self.options.acq.ref_point = torch.min(
                weighted_points, dim=0
            ).values.tolist()

        # get list of constraining functions
        constraint_callables = create_constraint_callables(self.vocs)

        acq = qNoisyExpectedHypervolumeImprovement(
            model,
            X_baseline=inputs,
            prune_baseline=True,
            constraints=constraint_callables,
            ref_point=self.options.acq.ref_point,
            sampler=self.sampler,
            objective=self.objective,
        )

        return acq
