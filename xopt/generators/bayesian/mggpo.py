from typing import Optional, Callable

import pandas as pd
import torch
from botorch.acquisition.multi_objective import MCMultiOutputObjective
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from pydantic import Field

from xopt.generators.bayesian.objectives import create_mobo_objective
from xopt.generators.ga.cnsga import CNSGAGenerator
from .bayesian_generator import MultiObjectiveBayesianGenerator


class MGGPOGenerator(MultiObjectiveBayesianGenerator):
    """
    Multi-Generation Gaussian Process Optimization (MGGPO) generator.
    Combines multi-objective bayesian optimization with genetic algorithms
    to do highly-parallelized multi-objective optimization.

    Attributes
    ----------
    name : str
        The name of the generator.
    population_size : int
        The population size for the genetic algorithm.
    supports_batch_generation : bool
        Indicates if the generator supports batch candidate generation.
    ga_generator : Optional[CNSGAGenerator]
        The CNSGA generator used to generate candidates.

    Methods
    -------
    propose_candidates(self, model: torch.nn.Module, n_candidates: int = 1) -> torch.Tensor
        Propose candidates for Bayesian Optimization.
    add_data(self, new_data: pd.DataFrame)
        Add new data to the generator.
    get_acquisition(self, model: torch.nn.Module) -> Callable
        Get the acquisition function for Bayesian Optimization.
    _get_objective(self) -> Callable
        Create the multi-objective Bayesian optimization objective.
    _get_acquisition(self, model: torch.nn.Module) -> qLogNoisyExpectedHypervolumeImprovement
        Create the Log Expected Hypervolume Improvement acquisition function.
    """

    name = "mggpo"
    population_size: int = Field(64, description="population size for ga")
    supports_batch_generation: bool = True
    supports_constraints: bool = True

    ga_generator: Optional[CNSGAGenerator] = Field(
        None, description="CNSGA generator used to generate candidates"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # create GA generator
        self.ga_generator = CNSGAGenerator(
            vocs=self.vocs,
            population_size=self.population_size,
        )

    def propose_candidates(
        self, model: torch.nn.Module, n_candidates: int = 1
    ) -> torch.Tensor:
        """
        Propose candidates for Bayesian Optimization.

        Parameters
        ----------
        model : torch.nn.Module
            The model used for Bayesian Optimization.
        n_candidates : int, optional
            The number of candidates to propose, by default 1.

        Returns
        -------
        torch.Tensor
            The proposed candidates.

        Raises
        ------
        RuntimeError
            If not enough unique solutions are generated by the GA.
        """
        ga_candidates = self.ga_generator.generate(n_candidates * 10)
        ga_candidates = pd.DataFrame(ga_candidates)[self.vocs.variable_names].to_numpy()
        ga_candidates = torch.unique(
            torch.tensor(ga_candidates, **self.tkwargs), dim=0
        ).reshape(-1, 1, self.vocs.n_variables)

        if ga_candidates.shape[0] < n_candidates:
            raise RuntimeError("not enough unique solutions generated by the GA!")

        acq_funct = self.get_acquisition(self.model)
        acq_funct_vals = acq_funct(ga_candidates)
        best_idxs = torch.argsort(acq_funct_vals, descending=True)[:n_candidates]

        candidates = ga_candidates[best_idxs]
        return candidates.reshape(n_candidates, self.vocs.n_variables)

    def add_data(self, new_data: pd.DataFrame):
        """
        Add new data to the generator.

        Parameters
        ----------
        new_data : pd.DataFrame
            The new data to be added.
        """
        super().add_data(new_data)
        self.ga_generator.add_data(self.data)

    def get_acquisition(self, model: torch.nn.Module) -> Callable:
        """
        Get the acquisition function for Bayesian Optimization.

        Parameters
        ----------
        model : torch.nn.Module
            The model used for Bayesian Optimization.

        Returns
        -------
        Callable
            The acquisition function.
        """
        # TODO: add error if fixed features - why is this not supported?
        if model is None:
            raise ValueError("model cannot be None")

        # get base acquisition function
        acq = self._get_acquisition(model)
        acq = acq.to(**self.tkwargs)
        return acq

    def _get_objective(self) -> MCMultiOutputObjective:
        """
        Create the multi-objective Bayesian optimization objective.
        """
        return create_mobo_objective(self.vocs).to(**self.tkwargs)

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
        # get reference point from data
        inputs = self.get_input_data(self.data)
        sampler = self._get_sampler(model)

        acq = qLogNoisyExpectedHypervolumeImprovement(
            model,
            X_baseline=inputs,
            prune_baseline=True,
            constraints=self._get_constraint_callables(),
            ref_point=self.torch_reference_point,
            sampler=sampler,
            objective=self._get_objective(),
            cache_root=False,
        )

        return acq
