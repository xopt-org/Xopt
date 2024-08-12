from abc import ABC, abstractmethod
from typing import ClassVar, Dict, List, Tuple

import torch
from botorch.models.model import Model
from pydantic import Field, PositiveInt
from torch import Tensor

from xopt.pydantic import XoptBaseModel


class Algorithm(XoptBaseModel, ABC):
    name: ClassVar[str] = "base_algorithm"
    n_samples: PositiveInt = Field(
        default=20, description="number of execution paths to generate"
    )

    @abstractmethod
    def get_execution_paths(
        self, model: Model, bounds: Tensor
    ) -> Tuple[Tensor, Tensor, Dict]:
        pass

    @abstractmethod
    def evaluate_virtual_objective(
        self,
        model: Model,
        x: Tensor,
        bounds: Tensor,
        n_samples: int,
        tkwargs: dict = None,
    ) -> Tensor:
        """
        Evaluates virtual objective at inputs given by x.
        Inputs:
            x: tensor, shape `num_points x ndim`
        Returns:
            objective_values: tensor, shape `n_samples x num_points x 1`
        """
        pass


class GridScanAlgorithm(Algorithm, ABC):
    name = "grid_scan_algorithm"
    n_mesh_points: PositiveInt = Field(
        default=10, description="number of mesh points along each axis"
    )

    def create_mesh(self, bounds: Tensor):
        """utility function used to create mesh for evaluating posteriors on"""
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
    observable_names_ordered: List[str] = Field(
        default=["y1"],
        description="names of observable/objective models used in this algorithm",
    )
    minimize: bool = True

    def get_execution_paths(
        self, model: Model, bounds: Tensor
    ) -> Tuple[Tensor, Tensor, Dict]:
        """get execution paths that minimize the objective function"""

        # build evaluation mesh
        test_points = self.create_mesh(bounds).to(model.models[0].train_targets)

        # get samples of the model posterior at mesh points
        posterior_samples = self.evaluate_virtual_objective(
            model, test_points, bounds, self.n_samples
        )

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

        # collect secondary results in a dict
        results_dict = {
            "test_points": test_points,
            "posterior_samples": posterior_samples,
            "execution_paths": torch.hstack((x_opt, y_opt)),
            "solution_center": solution_center,
            "solution_entropy": solution_entropy,
        }

        # return execution paths
        return x_opt.unsqueeze(-2), y_opt.unsqueeze(-2), results_dict

    def evaluate_virtual_objective(
        self,
        model: Model,
        x: Tensor,
        bounds: Tensor,
        n_samples: int,
        tkwargs: dict = None,
    ) -> Tensor:
        """Evaluate virtual objective (samples)"""

        # get samples of the model posterior at inputs given by x
        with torch.no_grad():
            post = model.posterior(x)
            objective_values = post.rsample(torch.Size([n_samples]))

        return objective_values


class CurvatureGridOptimize(GridOptimize):
    use_mean: bool = False

    def evaluate_virtual_objective(
        self,
        model: Model,
        x: Tensor,
        bounds: Tensor,
        n_samples: int,
        tkwargs: dict = None,
    ) -> Tensor:
        """Evaluate virtual objective (samples)"""

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

        return objective_values
