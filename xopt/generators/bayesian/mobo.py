from typing import List

import torch
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement

from xopt import VOCS
from xopt.generators.bayesian import BayesianGenerator
from .objectives import create_mobo_objective, create_constraint_callables
from .options import BayesianOptions, AcqOptions


class MOBOOptions(AcqOptions):
    ref_point: List[float] = None
    use_data_as_reference: bool = True


class MOBOGenerator(BayesianGenerator):
    def __init__(self, vocs: VOCS, **kwargs):

        # create weighted mc objective according to vocs
        objective = create_mobo_objective(vocs)

        options = BayesianOptions(acq=MOBOOptions())
        options.acq.objective = objective

        super(MOBOGenerator, self).__init__(vocs, options, **kwargs)

    def _get_acquisition(self, model):
        # get reference point from data
        inputs, outputs = self.get_training_data(self.data)
        if self.options.acq.use_data_as_reference:
            weighted_points = self.options.acq.objective(outputs)
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
            **self.options.acq.dict()
        )

        return acq
