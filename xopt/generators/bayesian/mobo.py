import torch
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement

from xopt import VOCS
from xopt.generators.bayesian import BayesianGenerator
from .utils import create_mobo_objective, create_constraint_callables


class MOBOGenerator(BayesianGenerator):
    def __init__(self, vocs: VOCS, **kwargs):
        super(MOBOGenerator, self).__init__(vocs)

        # create weighted mc objective according to vocs
        objective = create_mobo_objective(vocs)

        # add default arguments for acqf
        self.options["acqf_kw"].update({
            "ref_point": None,
            "use_data_as_reference": True,
            "objective": objective,
        })

        self.options["optim_kw"].update({
            "sequential": True
        })

        self.options.update(kwargs)

    def get_acquisition(self, model):
        # get reference point from data
        inputs, outputs = self.get_training_data()
        if self.options["acqf_kw"]["use_data_as_reference"]:
            weighted_points = self.options["acqf_kw"]["objective"](outputs)
            self.options["acqf_kw"]["ref_point"] = torch.min(
                weighted_points, dim=0
            ).values.tolist()

        # get list of constraining functions
        constraint_callables = create_constraint_callables(self.vocs)

        acq = qNoisyExpectedHypervolumeImprovement(
            model,
            X_baseline=inputs,
            prune_baseline=True,
            constraints=constraint_callables,
            **self.options["acqf_kw"]
        )

        return acq
