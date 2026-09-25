import warnings

from botorch.acquisition import (
    ScalarizedPosteriorTransform,
    UpperConfidenceBound,
    qUpperConfidenceBound,
)
from gpytorch import Module
from pydantic import Field
import torch

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


class ShiftedAcquisitionFunction(torch.nn.Module):
    def __init__(self, base_acq, shift: float = 0.0):
        super().__init__()
        self.base_acq = base_acq
        self.shift = shift

    def forward(self, X):
        return self.base_acq(X) + self.shift


# TODO: is log necessary for numeric stability of constrained softplus case? need to benchmark
class UpperConfidenceBoundGenerator(BayesianGenerator):
    """
    Bayesian optimization generator using Log Expected Improvement.

    Attributes
    ----------
    beta : float
        Beta parameter for UCB optimization, controlling the trade-off between exploration
        and exploitation. Higher values of beta prioritize exploration.

    """

    name = "upper_confidence_bound"
    beta: float = Field(2.0, description="Beta parameter for UCB optimization")
    shift: float = Field(
        0.0,
        description="Vertical shift applied to the UCB acquisition function for use with constraints",
    )
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
shift : float, default 0.0
    Vertical shift applied to the UCB acquisition function for use with constraints.

Notes
-----
Using UCB with constraints may lead to invalid values if the base acquisition function
has negative values. The base acquisition function can be negative when objectives are
to be minimized OR maximized with negative values. In such cases, it is recommended
to set a positive shift value that is larger than the absolute value of the most
negative objective value to ensure non-negative acquisition values. Otherwise, the
acqusition function may produce uniformly zero values due to Softplus transformation.
    """ + formatted_base_docstring()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.vocs.n_constraints > 0:
            warnings.warn(
                "Using upper confidence bound with constraints may lead to invalid values "
                "if the base acquisition function has negative values. Use with "
                "caution. Please make sure to set a positive shift value to avoid "
                "non-negative values when using minimization or maximization problems "
                "with negative objective values.",
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

        # if constraints are present, shift ucb to be non-negative
        if self.vocs.n_constraints > 0:
            acq = ShiftedAcquisitionFunction(acq, shift=self.shift)

        return acq.to(**self.tkwargs)


class TDUpperConfidenceBoundGenerator(
    TimeDependentBayesianGenerator, UpperConfidenceBoundGenerator
):
    name = "time_dependent_upper_confidence_bound"
    __doc__ = """Implements Time-Dependent Bayesian Optimization using the Upper
            Confidence Bound acquisition function"""
