import pandas as pd
import torch
from botorch.acquisition import qExpectedImprovement

from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.custom_botorch.constrained_acqusition import (
    ConstrainedMCAcquisitionFunction,
)
from xopt.generators.bayesian.objectives import (
    create_constraint_callables,
    create_mc_objective,
)
from xopt.generators.bayesian.options import BayesianOptions
from xopt.vocs import VOCS


class ExpectedImprovementGenerator(BayesianGenerator):
    alias = "expected_improvement"

    def __init__(self, vocs: VOCS, options: BayesianOptions = None):
        """
        Generator using Expected improvement acquisition function

        Parameters
        ----------
        vocs: dict
            Standard vocs dictionary for xopt

        options: BayesianOptions
            Specific options for this generator
        """
        options = options or BayesianOptions()
        if not isinstance(options, BayesianOptions):
            raise ValueError("options must be a `BayesianOptions` object")

        if vocs.n_objectives != 1:
            raise ValueError("vocs must have one objective for optimization")

        super().__init__(vocs, options)

    @staticmethod
    def default_options() -> BayesianOptions:
        return BayesianOptions()

    def _get_objective(self):
        return create_mc_objective(self.vocs)

    def _get_acquisition(self, model):
        valid_data = self.data[
            pd.unique(self.vocs.variable_names + self.vocs.output_names)
        ].dropna()
        objective_data = self.vocs.objective_data(valid_data, "")

        best_f = torch.tensor(objective_data.max(), **self._tkwargs)

        qEI = qExpectedImprovement(
            model,
            best_f=best_f,
            sampler=self.sampler,
            objective=self.objective,
        )

        cqUCB = ConstrainedMCAcquisitionFunction(
            model, qEI, create_constraint_callables(self.vocs),
        )

        return cqUCB
