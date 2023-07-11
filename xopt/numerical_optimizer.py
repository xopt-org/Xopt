from abc import ABC, abstractmethod
from typing import ClassVar

import torch
from botorch.optim import optimize_acqf
from pydantic import Field, PositiveInt, field_validator, validator
from torch import Tensor

from xopt.pydantic import XoptBaseModel


class NumericalOptimizer(XoptBaseModel, ABC):
    name: ClassVar[str] = "base"

    class Config:
        extra = "forbid"

    @abstractmethod
    def optimize(self, function, bounds, n_candidates=1):
        """optimize a function to produce a number of candidate points that
        minimize the function"""
        pass


class LBFGSOptimizer(NumericalOptimizer):
    name: ClassVar[str] = "LBFGS"
    n_raw_samples: PositiveInt = Field(
        20,
        description="number of raw samples used to seed optimization",
    )
    n_restarts: PositiveInt = Field(
        20, description="number of restarts during acquistion function optimization"
    )
    max_iter: PositiveInt = Field(2000)

    class Config:
        validate_assignment = True

    @field_validator("n_restarts")
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
            options={"maxiter": self.max_iter},
        )
        return candidates


class GridOptimizer(NumericalOptimizer):
    """
    Numerical optimizer that uses a brute-force grid search to find the optimium.

    Parameters
    ----------
    n_grid_points: PositiveInt, optional
        Number of mesh points per axis to sample. Algorithm time scales as
        `n_grd_points`^`input_dimension`

    """

    name: ClassVar[str] = "grid"
    n_grid_points: PositiveInt = Field(
        10, description="number of grid points per axis used for optimization"
    )

    def optimize(self, function, bounds, n_candidates=1):
        assert isinstance(bounds, Tensor)
        # create mesh
        if len(bounds) != 2:
            raise ValueError("bounds must have the shape [2, ndim]")

        dim = len(bounds[0])
        # add in a machine eps
        eps = 1e-5
        linspace_list = [
            torch.linspace(
                bounds.T[i][0] + eps, bounds.T[i][1] - eps, self.n_grid_points
            )
            for i in range(dim)
        ]

        xx = torch.meshgrid(*linspace_list, indexing="ij")
        mesh_pts = torch.stack(xx).flatten(start_dim=1).T

        # evaluate the function on grid points
        f_values = function(mesh_pts.unsqueeze(1))

        # get the best n_candidates
        _, indicies = torch.sort(f_values)
        x_min = mesh_pts[indicies.squeeze().flipud()]
        return x_min[:n_candidates]
