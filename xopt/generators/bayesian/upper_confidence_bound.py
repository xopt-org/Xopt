import warnings

from botorch.acquisition import (
    AcquisitionFunction,
    ScalarizedPosteriorTransform,
    UpperConfidenceBound,
    qUpperConfidenceBound,
)
from gpytorch import Module
from pydantic import Field

from xopt.generators.bayesian.bayesian_generator import (
    BayesianGenerator,
    formatted_base_docstring,
)
from xopt.generators.bayesian.custom_botorch.constrained_acquisition import (
    ConstrainedMCAcquisitionFunction,
)
from xopt.generators.bayesian.custom_botorch.log_acquisition_function import (
    LogAcquisitionFunction,
)
from xopt.generators.bayesian.objectives import CustomXoptObjective
from xopt.generators.bayesian.time_dependent import TimeDependentBayesianGenerator
from xopt.generators.bayesian.turbo import (
    OptimizeTurboController,
    SafetyTurboController,
)
from xopt.generators.bayesian.utils import set_botorch_weights


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

    def get_acquisition(self, model: Module) -> AcquisitionFunction:
        if model is None:
            raise ValueError("model cannot be None")

        # get base acquisition function
        acq = self._get_acquisition(model)

        # TODO: is log necessary? can do just softplus - logei paper did not try logucb
        if len(self.vocs.constraints):
            try:
                sampler = acq.sampler
            except AttributeError:
                sampler = self._get_sampler(model)

            acq = ConstrainedMCAcquisitionFunction(
                model, acq, self._get_constraint_callables(), sampler=sampler
            )

            # log transform the result to handle the constraints
            acq = LogAcquisitionFunction(acq)
        else:
            # TODO: if no constraints, still do log_softplus to match constrained case? need to bench
            # acq = LogAcquisitionFunction(acq)
            pass

        acq = self._apply_fixed_features(acq)
        acq = acq.to(**self.tkwargs)
        return acq

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
