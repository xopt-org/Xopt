import torch
from botorch.acquisition import qExpectedImprovement

from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.custom_botorch.constrained_acqusition import (
    ConstrainedMCAcquisitionFunction,
)


class ExpectedImprovementGenerator(BayesianGenerator):
    name = "expected_improvement"
    supports_batch_generation = True
    __doc__ = """Implements Bayeisan Optimization using the Expected Improvement
        acquisition function"""

    def _get_acquisition(self, model):
        objective_data = self.vocs.objective_data(self.data, "").dropna()
        best_f = -torch.tensor(objective_data.min(), **self._tkwargs)
        sampler = self._get_sampler(model)

        qEI = qExpectedImprovement(
            model,
            best_f=best_f,
            sampler=sampler,
            objective=self._get_objective(),
        )

        cqEI = ConstrainedMCAcquisitionFunction(
            model,
            qEI,
            self._get_constraint_callables(),
        )

        return cqEI
