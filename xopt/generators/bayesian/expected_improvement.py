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
from xopt.generators.bayesian.objectives import CustomXoptObjective
from xopt.generators.bayesian.time_dependent import TimeDependentBayesianGenerator
from xopt.generators.bayesian.utils import set_botorch_weights


class ExpectedImprovementGenerator(BayesianGenerator):
    name = "expected_improvement"
    supports_batch_generation: bool = True

    __doc__ = (
        "Bayesian optimization generator using Expected improvement\n"
        + formatted_base_docstring()
    )

    def _get_acquisition(self, model):
        objective = self._get_objective()
        best_f = self._get_best_f(self.data, objective)

        if self.n_candidates > 1 or isinstance(objective, CustomXoptObjective):
            # MC sampling for generating multiple candidate points
            sampler = self._get_sampler(model)
            acq = qExpectedImprovement(
                model,
                best_f=best_f,
                sampler=sampler,
                objective=objective,
            )
        else:
            # analytic acquisition function for single candidate generation with
            # basic objective
            # note that the analytic version cannot handle custom objectives
            weights = set_botorch_weights(self.vocs).to(**self._tkwargs)
            posterior_transform = ScalarizedPosteriorTransform(weights)
            acq = ExpectedImprovement(
                model, best_f=best_f, posterior_transform=posterior_transform
            )

        return acq

    def _get_best_f(self, data, objective):
        """get best function value for EI based on the objective"""
        if isinstance(objective, CustomXoptObjective):
            best_f = objective(
                torch.tensor(
                    self.vocs.observable_data(data).to_numpy(), **self._tkwargs
                )
            ).max()
        else:
            # analytic acquisition function for single candidate generation
            best_f = -torch.tensor(
                self.vocs.objective_data(data).min().values, **self._tkwargs
            )

        return best_f


class TDExpectedImprovementGenerator(
    TimeDependentBayesianGenerator, ExpectedImprovementGenerator
):
    name = "time_dependent_expected_improvement"
    __doc__ = """Implements Time-Dependent Bayesian Optimization using the Expected
    Improvement acquisition function"""
