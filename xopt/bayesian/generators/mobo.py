import logging
from functools import partial

import torch
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning

# Logger
logger = logging.getLogger(__name__)


class MOBOGenerator:
    def __init__(self, ref,
                 batch_size=1,
                 sigma=None,
                 sampler=None,
                 num_restarts=20,
                 raw_samples=1024):

        self.ref = ref
        self._corrected_ref = None

        self.batch_size = batch_size
        self.sigma = sigma
        self.sampler = sampler
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

    def generate(self, model, bounds, vocs, **tkwargs):
        """Optimizes the qEHVI acquisition function and returns new candidate(s)."""
        n_obectives = len(vocs['objectives'])
        n_constraints = len(vocs['constraints'])

        self.ref = self.ref.to(tkwargs['device']) if isinstance(self.ref, torch.Tensor) else torch.tensor(self.ref,
                                                                                                          **tkwargs)
        self._corrected_ref = self.get_corrected_ref(self.ref, vocs)

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
            # use observations that are better than the specified reference point and feasible
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
            # define constraint function - see botorch docs for info - I'm not sure how it works
            constraints=constraint_functions
        )

        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=self.batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,  # used for initialization heuristic
            options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
            sequential=True,
        )

        return candidates.detach()

    @staticmethod
    def get_corrected_ref(ref, vocs):
        new_ref = ref.clone()
        for j, name in zip(range(len(vocs['objectives'])), vocs['objectives'].keys()):
            if vocs['objectives'][name] == 'MINIMIZE':
                new_ref[j] = -new_ref[j]
        return new_ref
