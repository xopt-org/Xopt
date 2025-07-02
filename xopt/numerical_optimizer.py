from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.optim.parameter_constraints import nonlinear_constraint_is_feasible

from pydantic import ConfigDict, Field, PositiveFloat, PositiveInt
from torch import Tensor

from xopt.pydantic import XoptBaseModel


class NumericalOptimizer(XoptBaseModel, ABC):
    """
    Base class for numerical optimizers.

    Attributes
    ----------
    name : str
        The name of the optimizer. Default is "base_numerical_optimizer".
    model_config : ConfigDict
        Configuration dictionary with extra fields forbidden.

    Methods
    -------
    optimize(function, bounds, n_candidates=1, **kwargs)
        Abstract method to optimize a function to produce a number of candidate points that minimize the function.
    """

    @abstractmethod
    def optimize(
        self, function: AcquisitionFunction, bounds: Tensor, n_candidates=1, **kwargs
    ):
        """Optimize a function to produce a number of candidate points that minimize the function."""
        pass


class LBFGSOptimizer(NumericalOptimizer):
    """
    LBFGSOptimizer is a numerical optimizer that uses the Limited-memory Broyden–Fletcher–Goldfarb–Shanno (LBFGS) algorithm.

    Attributes
    ----------
    n_restarts : PositiveInt
        Number of restarts during acquisition function optimization, default is 20.
    max_iter : PositiveInt
        Maximum number of iterations for the optimizer, default is 2000.
    max_time : Optional[PositiveFloat]
        Maximum time allowed for optimization, default is None (no time limit).

    Methods
    -------
    optimize(function, bounds, n_candidates=1, **kwargs)
        Optimize the given acquisition function within the specified bounds.

    Parameters
    ----------
    function : callable
        The acquisition function to be optimized.
    bounds : Tensor
        The bounds within which to optimize the acquisition function. Must have shape [2, ndim].
    n_candidates : int, optional
        Number of candidates to return, default is 1.
    **kwargs : dict
        Additional keyword arguments to pass to the optimizer.

    Returns
    -------
    candidates : Tensor
        The optimized candidates.
    """

    name: str = Field("LBFGS", frozen=True)
    n_restarts: PositiveInt = Field(
        20, description="number of restarts during acquisition function optimization"
    )
    max_iter: PositiveInt = Field(
        2000, description="maximum number of optimization steps"
    )
    max_time: Optional[PositiveFloat] = Field(
        5.0, description="maximum time for optimization in seconds"
    )

    model_config = ConfigDict(validate_assignment=True)

    def optimize(self, function, bounds, n_candidates=1, **kwargs):
        """
        Optimize the given acquisition function within the specified bounds.

        Parameters
        ----------
        function : Callable
            The acquisition function to be optimized.
        bounds : Tensor
            A tensor specifying the bounds for the optimization. It must have the shape [2, ndim].
        n_candidates : int, optional
            The number of candidates to generate (default is 1).
        **kwargs : dict
            Additional keyword arguments to be passed to the acquisition function optimizer.

        Returns
        -------
        candidates : Tensor
            The optimized candidates.
        """

        assert isinstance(bounds, Tensor)
        if len(bounds) != 2:
            raise ValueError("bounds must have the shape [2, ndim]")

        # emperical testing showed that the max time is overrun slightly on the botorch side
        # fix by slightly reducing the max time passed to this function
        if self.max_time is not None:
            max_time = self.max_time * 0.8 - 0.01
        else:
            max_time = None

        candidates, _ = optimize_acqf(
            acq_function=function,
            bounds=bounds,
            q=n_candidates,
            raw_samples=self.n_restarts,
            num_restarts=self.n_restarts,
            timeout_sec=max_time,
            options={"maxiter": self.max_iter},
            **kwargs,
        )
        return candidates


class GridOptimizer(NumericalOptimizer):
    """
    Numerical optimizer that uses a brute-force grid search to find the optimum.

    Attributes
    ----------
    name : str
        The name of the optimizer. Default is "grid".
    n_grid_points : PositiveInt
        Number of mesh points per axis to sample. Algorithm time scales as `n_grid_points`^`input_dimension`.

    Methods
    -------
    optimize(function, bounds, n_candidates=1)
        Optimize the given acquisition function within the specified bounds.

    Parameters
    ----------
    function : callable
        The acquisition function to be optimized.
    bounds : Tensor
        The bounds within which to optimize the acquisition function. Must have shape [2, ndim].
    n_candidates : int, optional
        Number of candidates to return, default is 1.

    Returns
    -------
    candidates : Tensor
        The optimized candidates.
    """

    name: str = Field("grid", frozen=True)
    n_grid_points: PositiveInt = Field(
        10, description="number of grid points per axis used for optimization"
    )

    def optimize(
        self,
        function: Callable,
        bounds: Tensor,
        n_candidates: int = 1,
        nonlinear_inequality_constraints: (list[tuple[Callable, bool]] | None) = None,
    ):
        """
        Optimize the given acquisition function within the specified bounds.

        Parameters
        ----------
        function : Callable
            The acquisition function to be optimized.
        bounds : Tensor
            A tensor specifying the bounds for the optimization. It must have the shape [2, ndim].
        n_candidates : int, optional
            The number of candidates to generate (default is 1).
        nonlinear_inequality_constraints : Optional[list[Callable]]
            A list of callables representing the nonlinear inequality constraints. Each callable represents a
            constraint of the form callable(x) >= 0. Callable() takes in a one-dimensional tensor of shape `d`
            and returns a scalar.

        Returns
        -------
        candidates : Tensor
            The optimized candidates.
        """
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

        # Apply nonlinear constraints -- remove f_value point where constraints(x) does not satisfy the constraints
        if nonlinear_inequality_constraints is not None:
            mask = torch.ones(f_values.shape, dtype=torch.bool, device=f_values.device)
            for constraint in nonlinear_inequality_constraints:
                mask &= nonlinear_constraint_is_feasible(
                    constraint, True, mesh_pts.unsqueeze(1)
                )
            f_values = f_values[mask]

        # if no points are feasible, raise an error
        if f_values.numel() == 0:
            raise ValueError("No feasible points found")

        # get the best n_candidates
        _, indicies = torch.sort(f_values)
        x_min = mesh_pts[indicies.squeeze().flipud()]
        return x_min[:n_candidates]
