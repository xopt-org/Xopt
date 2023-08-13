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


class GridMinimize(GridScanAlgorithm):
    model_names_ordered: List[str] = Field(
        default=["y1"],
        description="names of observable/objective models used in this algorithm",
    )

    def get_execution_paths(
        self, model: Model, bounds: Tensor
    ) -> Tuple[Tensor, Tensor, Dict]:
        """get execution paths that minimize the objective function"""

        # build evaluation mesh
        test_points = self.create_mesh(bounds).to(model.models[0].train_targets)

        # get samples of the model posterior at mesh points
        with torch.no_grad():
            post = model.posterior(test_points)
            post_samples = post.rsample(torch.Size([self.n_samples]))

        # get points that minimize each sample (execution paths)
        y_min, min_idx = torch.min(post_samples, dim=-2)
        min_idx = min_idx.squeeze()
        x_min = test_points[min_idx]

        # collect secondary results in a dict
        results_dict = {
            "test_points": test_points,
            "posterior_samples": post_samples,
            "execution_paths": torch.hstack((x_min, y_min)),
        }

        # return execution paths
        return x_min.unsqueeze(-2), y_min.unsqueeze(-2), results_dict
