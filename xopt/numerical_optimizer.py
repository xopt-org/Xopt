from abc import ABC, abstractmethod

import torch
from botorch.optim import optimize_acqf
from pydantic import Field, PositiveInt, validator
from torch import Tensor

from xopt.pydantic import XoptBaseModel


class NumericalOptimizer(XoptBaseModel, ABC):
    name: str = "base"

    class Config:
        extra = "forbid"

    @abstractmethod
    def optimize(self, function, bounds, n_candidates=1):
        """ optimize a function to produce a number of candidate points that
        minimize the function"""
        pass


class LBFGSOptimizer(NumericalOptimizer):
    name = "LBFGS"
    n_raw_samples: PositiveInt = Field(
        20,
        description="number of raw samples used to seed optimization",
    )
    n_restarts: PositiveInt = Field(
        20, description="number of restarts during acquistion function optimization"
    )

    class Config:
        validate_assignment = True

    @validator("n_restarts")
    def validate_num_restarts(cls, v: int, values):
        if v > values["n_raw_samples"]:
            raise ValueError(
                "num_restarts cannot be greater than number of " "raw_samples"
            )
        return v

    def optimize(self, function, bounds, n_candidates=1):
        assert isinstance(bounds, Tensor)
        if len(bounds) != 2:
            raise ValueError("bounds must have the shape [2, ndim]")
        candidates, out = optimize_acqf(
            acq_function=function,
            bounds=bounds,
            q=n_candidates,
            raw_samples=self.n_raw_samples,
            num_restarts=self.n_restarts,
        )
        return candidates


class GridOptimizer(NumericalOptimizer):
    name = "grid"
    n_grid_points: PositiveInt = Field(
        10, description="number of grid points per axis used for optimization"
    )

    def optimize(self, function, bounds, n_candidates=1):
        assert isinstance(bounds, Tensor)
        # create mesh
        if len(bounds) != 2:
            raise ValueError("bounds must have the shape [2, ndim]")

        dim = len(bounds[0])
        linspace_list = [
            torch.linspace(bounds.T[i][0], bounds.T[i][1], self.n_grid_points)
            for i in range(dim)
        ]

        xx = torch.meshgrid(*linspace_list, indexing="ij")
        mesh_pts = torch.stack(xx).flatten(start_dim=1).T

        # evaluate the function on grid points
        f_values = function(mesh_pts)

        # get the best n_candidates
        _, indicies = torch.sort(f_values)
        x_min = mesh_pts[indicies.squeeze()]
        return x_min[:n_candidates]
