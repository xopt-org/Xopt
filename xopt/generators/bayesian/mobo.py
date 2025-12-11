from typing import Optional

import torch
from botorch.acquisition.multi_objective import MCMultiOutputObjective
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.utils import draw_sobol_samples
from pydantic import Field
from torch import Tensor

from xopt.generators.bayesian.bayesian_generator import MultiObjectiveBayesianGenerator
from xopt.generators.bayesian.objectives import create_mobo_objective
from xopt.generators.bayesian.turbo import SafetyTurboController
from xopt.numerical_optimizer import LBFGSOptimizer


class MOBOGenerator(MultiObjectiveBayesianGenerator):
    """
    Implements Multi-Objective Bayesian Optimization using the Log Expected
    Hypervolume Improvement acquisition function.

    Attributes
    ----------
    name : str
        The name of the generator.
    supports_batch_generation : bool
        Indicates if the generator supports batch candidate generation.
    use_pf_as_initial_points : bool
        Flag to specify if Pareto front points are to be used during optimization
        of the acquisition function.

    Methods
    -------
    _get_objective(self) -> Callable
        Create the multi-objective Bayesian optimization objective.
    get_acquisition(self, model: torch.nn.Module) -> FixedFeatureAcquisitionFunction
        Get the acquisition function for Bayesian Optimization.
    _get_acquisition(self, model: torch.nn.Module) -> qLogNoisyExpectedHypervolumeImprovement
        Create the Log Expected Hypervolume Improvement acquisition function.
    _get_initial_conditions(self, n_candidates: int = 1) -> Optional[Tensor]
        Generate initial candidates for optimizing the acquisition function based on
        the Pareto front.
    """

    name = "mobo"
    supports_batch_generation: bool = True
    supports_constraints: bool = True
    use_pf_as_initial_points: bool = Field(
        False,
        description="flag to specify if pareto front points are to be used during "
        "optimization of the acquisition function",
    )
    __doc__ = """Implements Multi-Objective Bayesian Optimization using the Log Expected
            Hypervolume Improvement acquisition function"""

    _compatible_turbo_controllers = [SafetyTurboController]

    def _get_objective(self) -> MCMultiOutputObjective:
        """
        Create the multi-objective Bayesian optimization objective.
        """
        if self.custom_objective is not None:
            if self.vocs.n_objectives:
                raise RuntimeError(
                    "cannot specify objectives in VOCS "
                    "and a custom objective for the generator at the "
                    "same time"
                )

            objective = self.custom_objective
        else:
            objective = create_mobo_objective(self.vocs)

        return objective.to(**self.tkwargs)

    def get_acquisition(self, model: torch.nn.Module):
        """
        Get the acquisition function for Bayesian Optimization.
        Note that this needs to overwrite the base method due to
        how qLogExpectedHypervolumeImprovement handles constraints.

        Parameters
        ----------
        model : torch.nn.Module
            The model used for Bayesian Optimization.

        Returns
        -------
        FixedFeatureAcquisitionFunction
            The acquisition function.
        """
        if model is None:
            raise ValueError("model cannot be None")

        # get base acquisition function
        acq = self._get_acquisition(model)

        # apply fixed features if specified in the generator
        acq = self._apply_fixed_features(acq)

        acq = acq.to(**self.tkwargs)
        return acq

    def _get_acquisition(
        self, model: torch.nn.Module
    ) -> qLogNoisyExpectedHypervolumeImprovement:
        """
        Create the Log Expected Hypervolume Improvement acquisition function.

        Parameters
        ----------
        model : torch.nn.Module
            The model used for Bayesian Optimization.

        Returns
        -------
        qLogNoisyExpectedHypervolumeImprovement
            The Log Expected Hypervolume Improvement acquisition function.
        """
        inputs = self.get_input_data(self.data)
        sampler = self._get_sampler(model)

        acq = qLogNoisyExpectedHypervolumeImprovement(
            model,
            X_baseline=inputs,
            constraints=self._get_constraint_callables(),
            ref_point=self.torch_reference_point,
            sampler=sampler,
            objective=self._get_objective(),
            cache_root=False,
            prune_baseline=True,
        )
        return acq

    def _get_initial_conditions(self, n_candidates: int = 1) -> Optional[Tensor]:
        """
        Generate initial candidates for optimizing the acquisition function based on
        the Pareto front.

        If `use_pf_as_initial_points` flag is set to true then the current
        Pareto-optimal set is used as initial points for optimizing the acquisition
        function instead of randomly selected points (random points fill in the set
        if `num_restarts` is greater than the number of points in the Pareto set).

        Parameters
        ----------
        n_candidates : int, optional
            The number of candidates to generate, by default 1.

        Returns
        -------
        Optional[Tensor]
            A `num_restarts x q x d` tensor of initial conditions, or None if the
            Pareto front is not used.
        """
        if self.use_pf_as_initial_points:
            if isinstance(self.numerical_optimizer, LBFGSOptimizer):
                bounds = self._get_optimization_bounds()
                num_restarts = self.numerical_optimizer.n_restarts

                pf_locations, _, _, _ = self.get_pareto_front_and_hypervolume()

                # if there is no pareto front just return None to revert back to
                # default behavior
                if pf_locations is None:
                    return None

                initial_points = torch.clone(pf_locations)

                # add the q dimension
                initial_points = initial_points.unsqueeze(1)
                initial_points = initial_points.expand([-1, n_candidates, -1])

                # initial_points must equal the number of restarts
                if len(initial_points) < num_restarts:
                    # add random points to the list inside the bounds
                    sobol_samples = draw_sobol_samples(
                        bounds, num_restarts - len(initial_points), n_candidates
                    )

                    initial_points = torch.cat([initial_points, sobol_samples])
                elif len(initial_points) > num_restarts:
                    # if there are too many select the first `num_restarts` points at
                    # random
                    initial_points = initial_points[
                        torch.randperm(len(initial_points))
                    ][:num_restarts]

                return initial_points
            else:
                raise RuntimeWarning(
                    "cannot use PF as initial optimization points "
                    "for non-LBFGS optimizers, ignoring flag"
                )

        return None
