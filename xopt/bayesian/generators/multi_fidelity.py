import logging

import botorch.acquisition
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions

# Logger
from ..utils import get_bounds
from .generator import BayesianGenerator

logger = logging.getLogger(__name__)


class MultiFidelityGenerator(BayesianGenerator):
    def __init__(self, vocs,
                 batch_size=1,
                 fixed_cost=0.01,
                 target_fidelities=None,
                 acq=None,
                 num_restarts=20,
                 raw_samples=1024,
                 num_fantasies=128):

        super(MultiFidelityGenerator, self).__init__(vocs, batch_size, 1, num_restarts,raw_samples)

        self.num_fantasies = num_fantasies
        self.target_fidelities = target_fidelities
        self.cost_model = AffineFidelityCostModel(self.target_fidelities, fixed_cost)


        if acq is None:
            self.acq = PosteriorMean
        else:
            assert isinstance(acq, botorch.acquisition.AcquisitionFunction) or callable(acq)
            self.acq = acq

        self.cost = []

    def project(self, X):
        return project_to_target_fidelity(X=X, target_fidelities=self.target_fidelities)

    def get_mfkg(self, model, bounds, cost_aware_utility):

        curr_val_acqf = FixedFeatureAcquisitionFunction(
            acq_function=self.acq(model),
            d=len(self.vocs['variables']),
            columns=[len(self.vocs['variables']) - 1],
            values=[1],
        )

        _, current_value = optimize_acqf(
            acq_function=curr_val_acqf,
            bounds=bounds[:, :-1],
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options={"batch_limit": 10, "maxiter": 200},
        )

        return qMultiFidelityKnowledgeGradient(
            model=model,
            num_fantasies=self.num_fantasies,
            current_value=current_value,
            cost_aware_utility=cost_aware_utility,
            project=self.project,
        )

    def generate(self, model, **tkwargs):
        """

        Optimize Multifidelity acquisition function

        """
        assert list(self.vocs['variables'])[-1] == 'cost', 'last variable in vocs["variables"] must be "cost"'

        bounds = get_bounds(self.vocs, **tkwargs)

        cost_aware_utility = InverseCostWeightedUtility(cost_model=self.cost_model)

        X_init = gen_one_shot_kg_initial_conditions(
            acq_function=self.get_mfkg(model, bounds, cost_aware_utility),
            bounds=bounds,
            q=self.batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )

        candidates, _ = optimize_acqf(
            acq_function=self.get_mfkg(model, bounds, cost_aware_utility),
            bounds=bounds,
            q=self.batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            batch_initial_conditions=X_init,
            options={"batch_limit": 5, "maxiter": 200},
        )
        self.cost += [self.cost_model(candidates).sum()]
        return candidates.detach()

    def get_recommendation(self, model, **tkwargs):
        bounds = get_bounds(self.vocs, **tkwargs)
        rec_acqf = FixedFeatureAcquisitionFunction(
            acq_function=self.acq(model),
            d=len(self.vocs['variables']),
            columns=[len(self.vocs['variables']) - 1],
            values=[1],
        )

        final_rec, _ = optimize_acqf(
            acq_function=rec_acqf,
            bounds=bounds[:, :-1],
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options={"batch_limit": 5, "maxiter": 200},
        )

        return rec_acqf._construct_X_full(final_rec)
