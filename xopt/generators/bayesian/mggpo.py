from typing import Dict, List

import pandas as pd
import torch
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
from pydantic import Field

from xopt.generators.bayesian.objectives import (
    create_constraint_callables,
    create_mobo_objective,
)
from xopt.generators.ga.cnsga import CNSGAGenerator, CNSGAOptions

from xopt.vocs import VOCS
from .bayesian_generator import BayesianGenerator
from .options import AcqOptions, BayesianOptions


class MGGPOAcqOptions(AcqOptions):
    ref_point: List[float] = Field(
        None, description="reference point for multi-objective optimization"
    )
    use_data_as_reference: bool = Field(
        True,
        description="flag to determine if the dataset determines the reference point",
    )
    n_ga_samples: int = Field(
        128, description="number of genetic algorithm samples to use"
    )


class MGGPOOptions(BayesianOptions):
    acq = MGGPOAcqOptions()


class MGGPOGenerator(BayesianGenerator):
    alias = "mggpo"

    def __init__(self, vocs: VOCS, options: MGGPOOptions = MGGPOOptions()):
        if not isinstance(options, MGGPOOptions):
            raise ValueError("options must be a MGGPOOptions object")

        super().__init__(vocs, options)

        # create GA generator
        self.ga_generator = CNSGAGenerator(
            vocs,
            options=CNSGAOptions(population_size=self.options.acq.n_ga_samples),
        )

    @staticmethod
    def default_options() -> MGGPOOptions:
        return MGGPOOptions()

    def generate(self, n_candidates: int) -> List[Dict]:
        if self.data.empty:
            return self.vocs.random_inputs(self.options.n_initial)
        else:

            if n_candidates > self.options.acq.n_ga_samples:
                raise ValueError(
                    "n_candidates must be less than or equal to n_ga_samples"
                )
            ga_candidates = self.ga_generator.generate(self.options.acq.n_ga_samples)
            ga_candidates = pd.DataFrame(ga_candidates)[
                self.vocs.variable_names
            ].to_numpy()
            ga_candidates = torch.tensor(ga_candidates).reshape(
                -1, 1, self.vocs.n_variables
            )

            # evaluate the acquisition function on the ga candidates
            self.train_model()
            acq_funct = self.get_acquisition(self.model)
            acq_funct_vals = acq_funct(ga_candidates)
            best_idxs = torch.argsort(acq_funct_vals, descending=True)[:n_candidates]

            candidates = ga_candidates[best_idxs]
            return self.vocs.convert_numpy_to_inputs(
                candidates.reshape(n_candidates, self.vocs.n_variables).numpy()
            )

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
