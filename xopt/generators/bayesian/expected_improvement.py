import torch
from botorch.acquisition import (
    ExpectedImprovement,
    qExpectedImprovement,
    ScalarizedPosteriorTransform,
)

from xopt.generators.bayesian.bayesian_generator import (
    BayesianGenerator,
    formatted_base_docstring,
)
from xopt.generators.bayesian.utils import set_botorch_weights


class ExpectedImprovementGenerator(BayesianGenerator):
    name = "expected_improvement"
    supports_batch_generation: bool = True

    __doc__ = (
        "Bayesian optimization generator using Expected improvement\n"
        + formatted_base_docstring()
    )

    def _get_acquisition(self, model):
        objective_data = self.vocs.objective_data(self.data, "").dropna()
        best_f = -torch.tensor(objective_data.min().values, **self._tkwargs)

        if self.n_candidates > 1:
            # MC sampling for generating multiple candidate points
            sampler = self._get_sampler(model)
            acq = qExpectedImprovement(
                model,
                best_f=best_f,
                sampler=sampler,
                objective=self._get_objective(),
            )
        else:
            # analytic acquisition function for single candidate generation
            weights = set_botorch_weights(self.vocs).to(**self._tkwargs)
            posterior_transform = ScalarizedPosteriorTransform(weights)
            acq = ExpectedImprovement(
                model, best_f=best_f, posterior_transform=posterior_transform
            )

        return acq
