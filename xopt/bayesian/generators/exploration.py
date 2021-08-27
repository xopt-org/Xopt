import logging
from functools import partial

import torch
from botorch.acquisition import GenericMCObjective

from .generator import BayesianGenerator
from ..acquisition.exploration import qBayesianExploration, BayesianExploration
from ..utils import UnsupportedError

# Logger
logger = logging.getLogger(__name__)


class BayesianExplorationGenerator(BayesianGenerator):
    def __init__(self, vocs,
                 batch_size=1,
                 sigma=None,
                 mc_samples=512,
                 num_restarts=20,
                 raw_samples=1024,
                 use_gpu=False):

        optimize_options = {'sequential': True}
        super(BayesianExplorationGenerator,
              self).__init__(vocs,
                             self.create_acq,
                             batch_size,
                             num_restarts,
                             raw_samples,
                             mc_samples=mc_samples,
                             optimize_options=optimize_options,
                             use_gpu=use_gpu)

        if batch_size != 1 and sigma is not None:
            raise UnsupportedError("`not possible to use proximal term in "
                                   "multi-batch setting")
        self.sigma = sigma

    def create_acq(self, model):
        """

        Optimize Bayesian Exploration

        model should be a SingleTaskGP model trained such that the output has a shape
        n x m + 1 where the first element is the target function for exploration and
        m is the number of constraints

        """
        n_constraints = len(self.vocs['constraints'])
        n_variables = len(self.vocs['variables'])

        # serialized Bayesian Exploration
        if self.optimize_options['q'] == 1:
            if self.sigma is None:
                self.sigma = torch.eye(n_variables, **self.tkwargs) * 1e10

            elif not isinstance(self.sigma, torch.Tensor):
                self.sigma = torch.tensor(self.sigma.copy(), **self.tkwargs)

            constraint_dict = {}
            for i in range(1, n_constraints + 1):
                constraint_dict[i] = [None, 0.0]

            constraint_dict = constraint_dict if len(constraint_dict) else None
            acq_func = BayesianExploration(model, 0, constraint_dict, self.sigma)

        # batched Bayesian Exploration
        else:
            mc_obj = GenericMCObjective(lambda Z, X: Z[..., 0])

            # define constraint functions - note issues with lambda implementation
            # https://tinyurl.com/j8wmckd3
            def constr_func(Z, index=-1):
                return Z[..., index]

            constraint_functions = []
            for i in range(1, n_constraints + 1):
                constraint_functions += [partial(constr_func, index=-i)]

            acq_func = qBayesianExploration(model, self.sampler, mc_obj,
                                            constraints=constraint_functions)

        return acq_func
