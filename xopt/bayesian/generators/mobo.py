import logging
from functools import partial
from typing import Dict, Optional, Union, List

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective.monte_carlo import \
    qExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.models.model import Model
from botorch.utils.multi_objective.box_decompositions.non_dominated import \
    NondominatedPartitioning
from torch import Tensor

from .generator import BayesianGenerator

# Logger

logger = logging.getLogger(__name__)


class MOBOGenerator(BayesianGenerator):
    def __init__(self,
                 vocs: [Dict],
                 ref: Union[Tensor, List],
                 batch_size: Optional[int] = 1,
                 sigma: Optional[Union[Tensor, List]] = None,
                 mc_samples: Optional[int] = 512,
                 num_restarts: Optional[int] = 20,
                 raw_samples: Optional[int] = 1024,
                 use_gpu: Optional[bool] = False) -> None:

        self.ref = ref
        self._corrected_ref = None
        self.sigma = sigma
        optimize_options = {'sequential': True}

        super(MOBOGenerator, self).__init__(vocs,
                                            self.create_acqf,
                                            batch_size,
                                            num_restarts,
                                            raw_samples,
                                            mc_samples=mc_samples,
                                            optimize_options=optimize_options,
                                            use_gpu=use_gpu)

    def create_acqf(self, model: [Model]) -> AcquisitionFunction:
        n_obectives = len(self.vocs['objectives'])
        n_constraints = len(self.vocs['constraints'])

        self.ref = (self.ref.to(self.tkwargs['device'])
                    if isinstance(self.ref, torch.Tensor) else
                    torch.tensor(self.ref, **self.tkwargs))

        self._corrected_ref = self.get_corrected_ref(self.ref)

        train_outputs = model.train_targets.T
        train_y = train_outputs[:, :n_obectives]
        train_c = train_outputs[:, n_obectives:]

        # compute feasible observations
        is_feas = (train_c <= 0).all(dim=-1)

        # compute points that are better than the known reference point
        better_than_ref = (train_y > self._corrected_ref).all(dim=-1)
        # partition non-dominated space into disjoint rectangles
        partitioning = NondominatedPartitioning(
            ref_point=self._corrected_ref,
            # use observations that are better than the specified reference point and
            # feasible
            Y=train_y[better_than_ref & is_feas],

        )

        # define constraint functions - note issues with lambda implementation
        # https://tinyurl.com/j8wmckd3
        def constr_func(Z, index=-1):
            return Z[..., index]

        constraint_functions = []
        for i in range(1, n_constraints + 1):
            constraint_functions += [partial(constr_func, index=-i)]

        acq_func = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=self._corrected_ref.tolist(),  # use known reference point
            partitioning=partitioning,
            # define an objective that specifies which outcomes are the objectives
            objective=IdentityMCMultiOutputObjective(outcomes=list(range(n_obectives))),
            # define constraint function - see botorch docs for info - I'm not sure
            # how it works
            constraints=constraint_functions,
            sampler=self.sampler
        )

        return acq_func

    def get_corrected_ref(self, ref):
        new_ref = ref.clone()
        for j, name in zip(range(len(self.vocs['objectives'])),
                           self.vocs['objectives'].keys()):
            if self.vocs['objectives'][name] == 'MINIMIZE':
                new_ref[j] = -new_ref[j]
        return new_ref
