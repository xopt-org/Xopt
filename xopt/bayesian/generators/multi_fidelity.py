import logging
import copy

import torch
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean, AcquisitionFunction
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions

from ...vocs_tools import get_bounds
from .generator import BayesianGenerator

# Logger
logger = logging.getLogger(__name__)


class MultiFidelityGenerator(BayesianGenerator):
    def __init__(
        self,
        vocs,
        batch_size=1,
        fixed_cost=0.01,
        target_fidelities=None,
        base_acq=None,
        num_restarts=20,
        raw_samples=1024,
        mc_samples=512,
        num_fantasies=128,
        use_gpu=False,
    ):

        super(MultiFidelityGenerator, self).__init__(
            vocs,
            self.create_acq,
            batch_size,
            num_restarts,
            raw_samples,
            mc_samples,
            use_gpu=use_gpu,
        )

        if list(self.vocs.variables)[-1] != "cost":
            raise ValueError("`cost` must be the last keyword in vocs[variables]")

        self.num_fantasies = num_fantasies
        self.target_fidelities = target_fidelities
        self.cost_model = AffineFidelityCostModel(self.target_fidelities, fixed_cost)
        self.X_pending = None

        if base_acq is None:
            self.base_acq = PosteriorMean
        else:
            # check to make sure acq_function is correct type
            if not (isinstance(base_acq, AcquisitionFunction) or callable(base_acq)):
                raise ValueError(
                    "`acq_func` is not type AcquisitionFunction or callable"
                )

            self.base_acq = base_acq

    def project(self, X):
        return project_to_target_fidelity(X=X, target_fidelities=self.target_fidelities)

    def get_mfkg(self, model, bounds, cost_aware_utility):

        curr_val_acqf = FixedFeatureAcquisitionFunction(
            acq_function=self.base_acq(model),
            d=len(self.vocs.variables),
            columns=[len(self.vocs.variables) - 1],
            values=[1],
        )

        _, current_value = optimize_acqf(
            acq_function=curr_val_acqf,
            bounds=bounds[:, :-1],
            q=1,
            num_restarts=self.optimize_options["num_restarts"],
            raw_samples=self.optimize_options["raw_samples"],
            options=self.optimize_options["options"],
        )

        return qMultiFidelityKnowledgeGradient(
            model=model,
            num_fantasies=self.num_fantasies,
            current_value=current_value,
            cost_aware_utility=cost_aware_utility,
            project=self.project,
            X_pending=self.X_pending,
        )

    def create_acq(self, model):
        """

        Create Multifidelity acquisition function

        """

        bounds = torch.tensor(get_bounds(self.vocs), **self.tkwargs)

        cost_aware_utility = InverseCostWeightedUtility(cost_model=self.cost_model)

        # get optimization options
        one_shot_options = copy.deepcopy(self.optimize_options)
        if "batch_initial_conditions" in one_shot_options:
            one_shot_options.pop("batch_initial_conditions")

        X_init = gen_one_shot_kg_initial_conditions(
            acq_function=self.get_mfkg(model, bounds, cost_aware_utility),
            bounds=bounds,
            **one_shot_options
        )

        self.optimize_options["batch_initial_conditions"] = X_init

        return self.get_mfkg(model, bounds, cost_aware_utility)

    def get_recommendation(self, model):
        bounds = torch.tensor(get_bounds(self.vocs), **self.tkwargs)
        rec_acqf = FixedFeatureAcquisitionFunction(
            acq_function=self.base_acq(model),
            d=len(self.vocs.variables),
            columns=[len(self.vocs.variables) - 1],
            values=[1],
        )

        # get optimization options for recommendation optimization
        final_options = copy.deepcopy(self.optimize_options)
        for ele in ["q", "batch_initial_conditions"]:
            try:
                final_options.pop(ele)
            except KeyError:
                pass

        final_rec, _ = optimize_acqf(
            acq_function=rec_acqf, bounds=bounds[:, :-1], q=1, **final_options
        )

        return rec_acqf._construct_X_full(final_rec)
