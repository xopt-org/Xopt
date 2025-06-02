from pandas import DataFrame
import torch
from botorch.acquisition import (
    ScalarizedPosteriorTransform,
    LogExpectedImprovement,
    qLogExpectedImprovement,
    FixedFeatureAcquisitionFunction,
)

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


class ExpectedImprovementGenerator(BayesianGenerator):
    """
    Bayesian optimization generator using Log Expected Improvement.
    """

    name = "expected_improvement"
    supports_batch_generation: bool = True
    supports_single_objective: bool = True
    supports_constraints: bool = True

    __doc__ = (
        "Bayesian optimization generator using Expected improvement\n"
        + formatted_base_docstring()
    )

    _compatible_turbo_controllers = [OptimizeTurboController, SafetyTurboController]

    def get_acquisition(self, model):
        """
        Returns a function that can be used to evaluate the acquisition function.
        Overwrites base `get_acquisition` method.

        Parameters
        ----------
        model : Model
            The model used for Bayesian Optimization.

        Returns
        -------
        acq : AcquisitionFunction
            The acquisition function.
        """
        if model is None:
            raise ValueError("model cannot be None")

        # get base acquisition function
        acq = self._get_acquisition(model)

        # apply fixed features if specified in the generator
        if self.fixed_features is not None:
            # get input dim
            dim = len(self.model_input_names)
            columns = []
            values = []
            for name, value in self.fixed_features.items():
                columns += [self.model_input_names.index(name)]
                values += [value]

            acq = FixedFeatureAcquisitionFunction(
                acq_function=acq, d=dim, columns=columns, values=values
            )

        acq = acq.to(**self.tkwargs)
        return acq

    def _get_acquisition(self, model: torch.nn.Module):
        """
        Get the acquisition function for Bayesian Optimization.

        Parameters
        ----------
        model : Model
            The model used for Bayesian Optimization.

        Returns
        -------
        acq : AcquisitionFunction
            The acquisition function.
        """
        objective = self._get_objective()
        best_f = self._get_best_f(self.data, objective)

        if self.n_candidates > 1 or isinstance(objective, CustomXoptObjective):
            # MC sampling for generating multiple candidate points
            sampler = self._get_sampler(model)
            acq = qLogExpectedImprovement(
                model,
                best_f=best_f,
                sampler=sampler,
                objective=objective,
                constraints=self._get_constraint_callables(),
            )
        else:
            # analytic acquisition function for single candidate generation with
            # basic objective
            # note that the analytic version cannot handle custom objectives
            weights = set_botorch_weights(self.vocs).to(**self.tkwargs)
            posterior_transform = ScalarizedPosteriorTransform(weights)
            acq = LogExpectedImprovement(
                model, best_f=best_f, posterior_transform=posterior_transform
            )

        return acq

    def _get_best_f(self, data: DataFrame, objective: CustomXoptObjective):
        """
        Get the best function value for Expected Improvement based on the objective.

        Parameters
        ----------
        data : pd.DataFrame
            The data used for optimization.
        objective : Objective
            The objective function.

        Returns
        -------
        best_f : Tensor
            The best function value.
        """
        if isinstance(objective, CustomXoptObjective):
            best_f = objective(
                torch.tensor(self.vocs.observable_data(data).to_numpy(), **self.tkwargs)
            ).max()
        else:
            # analytic acquisition function for single candidate generation
            best_f = -torch.tensor(
                self.vocs.objective_data(data).min().values, **self.tkwargs
            )

        return best_f


class TDExpectedImprovementGenerator(
    TimeDependentBayesianGenerator, ExpectedImprovementGenerator
):
    name = "time_dependent_expected_improvement"
    __doc__ = (
        "Implements Time-Dependent Bayesian Optimization using the Expected "
        "Improvement acquisition function\n" + formatted_base_docstring()
    )
