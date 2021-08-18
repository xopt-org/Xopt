import logging

from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions

# Logger
from xopt.bayesian.utils import get_bounds

logger = logging.getLogger(__name__)


class MultiFidelityGenerator:
    def __init__(self,
                 batch_size=1,
                 sampler=None,
                 num_restarts=20,
                 raw_samples=1024,
                 num_fantasies=128):
        self.batch_size = batch_size
        self.sampler = sampler
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.num_fantasies = num_fantasies
        self.target_fidelities = None

        self.cost = []

    def project(self, X):
        assert self.target_fidelities is not None
        return project_to_target_fidelity(X=X, target_fidelities=self.target_fidelities)

    def get_mfkg(self, model, bounds, cost_aware_utility, vocs):

        curr_val_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(model),
            d=len(vocs['variables']),
            columns=[len(vocs['variables'])-1],
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

    def generate(self, model, vocs, **tkwargs):
        """

        Optimize Multifidelity acquisition function

        """
        assert list(vocs['variables'])[-1] == 'cost', 'last variable in vocs["variables"] must be "cost"'

        bounds = get_bounds(vocs, **tkwargs)

        self.target_fidelities = {len(vocs['variables']) - 1: 1.0}
        cost_model = AffineFidelityCostModel(fidelity_weights=self.target_fidelities, fixed_cost=5.0)
        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

        X_init = gen_one_shot_kg_initial_conditions(
            acq_function=self.get_mfkg(model, bounds, cost_aware_utility, vocs),
            bounds=bounds,
            q=self.batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )

        candidates, _ = optimize_acqf(
            acq_function=self.get_mfkg(model, bounds, cost_aware_utility, vocs),
            bounds=bounds,
            q=self.batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            batch_initial_conditions=X_init,
            options={"batch_limit": 5, "maxiter": 200},
        )
        self.cost += [cost_model(candidates).sum()]
        return candidates.detach()

    def get_recommendation(self, model, vocs, **tkwargs):
        bounds = get_bounds(vocs, **tkwargs)
        rec_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(model),
            d=len(vocs['variables']),
            columns=[len(vocs['variables'])-1],
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
