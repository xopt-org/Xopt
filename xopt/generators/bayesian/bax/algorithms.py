from abc import ABC, abstractmethod
from typing import ClassVar, List

import torch
from botorch.models.model import Model, ModelList
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, computed_field
from torch import Tensor

from xopt.pydantic import XoptBaseModel


class AlgorithmResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    input_execution_paths: Tensor
    output_execution_paths: Tensor


class OptimizationAlgorithmResult(AlgorithmResult):
    best_inputs: Tensor
    best_objective: Tensor


class VirtualMeasurementResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    objective: Tensor


class Algorithm(XoptBaseModel, ABC):
    """
    Base class for algorithms used in BAX.

    Attributes
    ----------
    name : ClassVar[str]
        The name of the algorithm.
    n_samples : PositiveInt
        Number of execution paths to generate.

    Methods
    -------
    get_execution_paths(self, model: Model, bounds: Tensor) -> Tuple[Tensor, Tensor, Dict]
        Get execution paths for the algorithm.
    perform_virtual_measurement(self, model: Model, x: Tensor, bounds: Tensor, n_samples: int, tkwargs: dict = None) -> Tensor
        Perform the virtual measurement and calculate objective values at the given inputs.
    """

    name: ClassVar[str] = "base_algorithm"
    n_samples: PositiveInt = Field(
        default=20, description="number of execution paths to generate"
    )

    @computed_field
    @property
    def class_path(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__name__}"

    @abstractmethod
    def execute(self, model: Model, bounds: Tensor) -> AlgorithmResult:
        """
        Execute the algorithm on samples and return results.

        Parameters
        ----------
        model : Model
            The model to use for generating execution paths.
        bounds : Tensor
            The bounds for the optimization.

        Returns
        -------
        AlgorithmResult
            The algorithm result with input and output execution paths.
        """
        pass

    @abstractmethod
    def perform_virtual_measurement(
        self,
        model: Model,
        x: Tensor,
        bounds: Tensor,
        n_samples: int,
        tkwargs: dict = None,
    ) -> VirtualMeasurementResult:
        """
        Evaluate the virtual objective at the given inputs.

        Parameters
        ----------
        model : Model
            The model to use for evaluating the virtual objective.
        x : Tensor
            The inputs at which to evaluate the virtual objective.
        bounds : Tensor
            The bounds for the optimization.
        n_samples : int
            The number of samples to generate.
        tkwargs : dict, optional
            Additional keyword arguments for the evaluation.

        Returns
        -------
        VirtualMeasurementResult
            The virtual measurement result with computed objective values.
        """
        pass  # pragma: no cover


class GridScanAlgorithm(Algorithm, ABC):
    """
    Grid scan algorithm for BAX.

    Attributes
    ----------
    name : str
        The name of the algorithm.
    n_mesh_points : PositiveInt
        Number of mesh points along each axis.

    Methods
    -------
    create_mesh(self, bounds: Tensor) -> Tensor
        Create a mesh for evaluating posteriors on.
    """

    name = "grid_scan_algorithm"
    n_mesh_points: PositiveInt = Field(
        default=10, description="number of mesh points along each axis"
    )

    def create_mesh(self, bounds: Tensor) -> Tensor:
        """
        Create a mesh for evaluating posteriors on.

        Parameters
        ----------
        bounds : Tensor
            The bounds for the optimization.

        Returns
        -------
        Tensor
            The mesh points.

        Raises
        ------
        ValueError
            If the bounds do not have the shape [2, ndim].
        """
        if len(bounds) != 2:
            raise ValueError("bounds must have the shape [2, ndim]")

        dim = len(bounds[0])
        linspace_list = [
            torch.linspace(bounds.T[i][0], bounds.T[i][1], self.n_mesh_points)
            for i in range(dim)
        ]

        xx = torch.meshgrid(*linspace_list, indexing="ij")
        mesh_pts = torch.stack(xx).flatten(start_dim=1).T

        return mesh_pts


class GridOptimize(GridScanAlgorithm):
    """
    Grid optimization algorithm for BAX.

    Attributes
    ----------
    observable_names_ordered : List[str]
        Names of observable/objective models used in this algorithm.
    minimize : bool
        Whether to minimize the objective function.

    Methods
    -------
    get_execution_paths(self, model: Model, bounds: Tensor) -> Tuple[Tensor, Tensor, Dict]
        Get execution paths that minimize the objective function.
    perform_virtual_measurement(self, model: Model, x: Tensor, bounds: Tensor, n_samples: int, tkwargs: dict = None) -> VirtualMeasurementResult
        Evaluate the virtual measurement and calculate objective values (samples).
    """

    observable_names_ordered: List[str] = Field(
        default=["y1"],
        description="names of observable/objective models used in this algorithm",
    )
    minimize: bool = True

    def execute(self, model: Model, bounds: Tensor) -> AlgorithmResult:
        """
        Execute the algorithm on samples and collect the results of the optimization.

        Parameters
        ----------
        model : Model
            The model to use for generating execution paths.
        bounds : Tensor
            The bounds for the optimization.

        Returns
        -------
        OptimizationAlgorithmResult
            Contains best_inputs, best_objective, input_execution_paths, output_execution_paths, and additional results.
        """
        # build evaluation mesh
        test_points = self.create_mesh(bounds)
        if isinstance(model, ModelList):
            test_points = test_points.to(model.models[0].train_targets)
        else:
            test_points = test_points.to(model.train_targets)

        # get samples of the model posterior at mesh points
        result = self.perform_virtual_measurement(
            model, test_points, bounds, self.n_samples
        )

        posterior_samples = result.objective
        # get points that minimize each sample (execution paths)
        if self.minimize:
            y_opt, opt_idx = torch.min(posterior_samples, dim=-2)
        else:
            y_opt, opt_idx = torch.max(posterior_samples, dim=-2)

        opt_idx = opt_idx.squeeze(dim=[-1])
        x_opt = test_points[opt_idx]

        # get the solution_center and solution_entropy for Turbo
        # note: the entropy calc here drops a constant scaling factor
        solution_center = x_opt.mean(dim=0).numpy()
        solution_entropy = float(torch.log(x_opt.std(dim=0) ** 2).sum())

        algorithm_result = OptimizationAlgorithmResult(
            best_inputs=x_opt,
            best_objective=y_opt,
            input_execution_paths=x_opt.unsqueeze(-2),
            output_execution_paths=y_opt.unsqueeze(-2),
            execution_paths=torch.hstack((x_opt, y_opt)),
            test_points=test_points,
            posterior_samples=posterior_samples,
            solution_center=solution_center,
            solution_entropy=solution_entropy,
        )
        # return algorithm result
        return algorithm_result

    def perform_virtual_measurement(
        self,
        model: Model,
        x: Tensor,
        bounds: Tensor,
        n_samples: int,
        tkwargs: dict = None,
    ) -> VirtualMeasurementResult:
        """
        Perform the virtual measurement (samples).

        Parameters
        ----------
        model : Model
            The model to use for evaluating the virtual objective.
        x : Tensor
            The inputs at which to evaluate the virtual objective.
        bounds : Tensor
            The bounds for the optimization.
        n_samples : int
            The number of samples to generate.
        tkwargs : dict, optional
            Additional keyword arguments for the evaluation.

        Returns
        -------
        VirtualMeasurementResult
            The virtual measurement result with calculated virtual objective values.
        """
        with torch.no_grad():
            post = model.posterior(x)
            objective_values = post.rsample(torch.Size([n_samples]))

        return VirtualMeasurementResult(objective=objective_values)


class CurvatureGridOptimize(GridOptimize):
    """
    Curvature grid optimization algorithm for BAX.

    Attributes
    ----------
    use_mean : bool
        Whether to use the mean of the posterior distribution.

    Methods
    -------
    perform_virtual_measurement(self, model: Model, x: Tensor, bounds: Tensor, n_samples: int, tkwargs: dict = None) -> VirtualMeasurementResult
        Perform the virtual measurement (samples) with curvature.
    """

    use_mean: bool = False

    def perform_virtual_measurement(
        self,
        model: Model,
        x: Tensor,
        bounds: Tensor,
        n_samples: int,
        tkwargs: dict = None,
    ) -> VirtualMeasurementResult:
        """
        Evaluate the virtual objective (samples) with curvature.

        Parameters
        ----------
        model : Model
            The model to use for evaluating the virtual objective.
        x : Tensor
            The inputs at which to evaluate the virtual objective.
        bounds : Tensor
            The bounds for the optimization.
        n_samples : int
            The number of samples to generate.
        tkwargs : dict, optional
            Additional keyword arguments for the evaluation.

        Returns
        -------
        VirtualMeasurementResult
            The virtual measurement result with calculated virtual objective values with curvature.
        """
        # get samples of the model posterior at inputs given by x
        with torch.no_grad():
            post = model.posterior(x)
            if self.use_mean:
                objective_values = post.mean.unsqueeze(0)
            else:
                objective_values = post.rsample(torch.Size([n_samples]))

        # pad sides with a single value on left and right
        # zero second order gradient at edges
        padding = (0, 0, 1, 1)  # e.g., padding with 1 value on both left and right
        objective_values = torch.nn.functional.pad(
            objective_values, padding, mode="replicate"
        )
        objective_values = torch.diff(objective_values, 2, dim=-2)
        objective_values[:, 0] = 0
        objective_values[:, -1] = 0

        return VirtualMeasurementResult(objective=objective_values)
