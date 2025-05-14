import warnings

from botorch.acquisition import (
    ScalarizedPosteriorTransform,
    UpperConfidenceBound,
    qUpperConfidenceBound,
)
from gpytorch import Module
from pydantic import Field

from xopt.errors import GeneratorWarning
from xopt.generators.bayesian.bayesian_generator import (
    BayesianGenerator,
    formatted_base_docstring,
)
from xopt.generators.bayesian.objectives import CustomXoptObjective
from xopt.generators.bayesian.time_dependent import TimeDependentBayesianGenerator
from xopt.generators.bayesian.turbo import (
    OptimizeTurboController,
    SafetyTurboController,
)
from xopt.generators.bayesian.utils import set_botorch_weights


# TODO: is log necessary for numeric stability of constrained softplus case? need to benchmark
class UpperConfidenceBoundGenerator(BayesianGenerator):
    name = "upper_confidence_bound"
    beta: float = Field(2.0, description="Beta parameter for UCB optimization")
    supports_batch_generation: bool = True
    supports_single_objective: bool = True
    supports_constraints: bool = True
    _compatible_turbo_controllers = [OptimizeTurboController, SafetyTurboController]

    __doc__ = """Bayesian optimization generator using Upper Confidence Bound

Attributes
----------
beta : float, default 2.0
    Beta parameter for UCB optimization, controlling the trade-off between exploration
    and exploitation. Higher values of beta prioritize exploration.

    """ + formatted_base_docstring()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.vocs.n_constraints > 0:
            warnings.warn(
                "Using upper confidence bound with constraints may lead to invalid values "
                "if the base acquisition function has negative values. Use with "
                "caution."
            )

    def propose_candidates(self, model: Module, n_candidates: int = 1):
        # TODO: convert to exception in the future
        if self.vocs.n_constraints > 0 and n_candidates > 1:
            warnings.warn(
                "Using UCB for constrained generation of multiple candidates is numerically unstable and "
                "will raise error in the future. Try expected improvement instead.",
                category=GeneratorWarning,
            )
        return super().propose_candidates(model, n_candidates)

    def _get_acquisition(self, model):
        objective = self._get_objective()
        if self.n_candidates > 1 or isinstance(objective, CustomXoptObjective):
            # MC sampling for generating multiple candidate points
            sampler = self._get_sampler(model)
            acq = qUpperConfidenceBound(
                model,
                sampler=sampler,
                objective=self._get_objective(),
                beta=self.beta,
            )
        else:
            # analytic acquisition function for single candidate generation
            weights = set_botorch_weights(self.vocs)
            posterior_transform = ScalarizedPosteriorTransform(weights)
            acq = UpperConfidenceBound(
                model, beta=self.beta, posterior_transform=posterior_transform
            )
        return acq.to(**self.tkwargs)


class TDUpperConfidenceBoundGenerator(
    TimeDependentBayesianGenerator, UpperConfidenceBoundGenerator
):
    name = "time_dependent_upper_confidence_bound"
    __doc__ = """Implements Time-Dependent Bayesian Optimization using the Upper
            Confidence Bound acquisition function"""
