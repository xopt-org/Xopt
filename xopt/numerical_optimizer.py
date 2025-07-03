from abc import ABC, abstractmethod
import time
from typing import Callable, Optional
import warnings

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.optim.parameter_constraints import nonlinear_constraint_is_feasible
from botorch.exceptions import CandidateGenerationError

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

    def optimize(
        self,
        function: Callable,
        bounds: Tensor,
        n_candidates: int = 1,
        nonlinear_inequality_constraints: (list[tuple[Callable, bool]] | None) = None,
        **kwargs,
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

        # if nonlinear constraints are provided, need to create an initial condition generator
        full_nonlinear_inequality_constraints = []
        if nonlinear_inequality_constraints is not None:
            # add in the boolean flag for the interpoint constraints
            for i, constraint in enumerate(nonlinear_inequality_constraints):
                full_nonlinear_inequality_constraints.append((constraint, True))

            warnings.warn(
                "Nonlinear inequality constraints are provided for LBFGS numerical optimization, "
                "using a random initial condition generator which may take a long time to sample enough points.",
                UserWarning,
            )
            ic_generator = get_random_ic_generator(
                full_nonlinear_inequality_constraints
            )

            kwargs["ic_generator"] = ic_generator
            kwargs["nonlinear_inequality_constraints"] = (
                full_nonlinear_inequality_constraints
            )

        try:
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

        except CandidateGenerationError:
            print(
                "Candidate generation failed, returning random valid samples which may not be optimal.",
            )
            return ic_generator(
                function,
                bounds,
                q=n_candidates,
                num_restarts=n_candidates,
                raw_samples=self.n_restarts,
            )[:n_candidates][0]


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
            mesh_pts = mesh_pts[mask]

            # if no points are feasible, raise an error
            if f_values.numel() == 0:
                raise ValueError("No feasible points found")

        # get the best n_candidates
        _, indicies = torch.sort(f_values)
        x_min = mesh_pts[indicies.squeeze().flipud()]
        return x_min[:n_candidates]


def get_random_ic_generator(
    nonlinear_constraints: list[Callable], max_resamples: int = 100
):
    """
    Get a random initial condition generator for the given nonlinear constraints.

    Parameters
    ----------
    nonlinear_constraints : list[Callable]
        A list of callables representing the nonlinear constraints. Each callable should take a tensor input
        and return a boolean mask indicating feasibility.
    max_resamples : int, optional
        Maximum number of resampling attempts to find valid initial conditions, default is 100. If no valid
        initial conditions are found after this many attempts, an error is raised.

    Returns
    -------
    random_ic_generator : Callable
        A callable that generates random initial conditions for the given function for use in botorch `optimize_acqf`.

    """

    def random_ic_generator(
        acq_function: AcquisitionFunction,
        bounds: Tensor,
        q: int,
        num_restarts: int,
        raw_samples: int,
        fixed_features: dict[int, float] | None = None,
        options: dict[str, bool | float | int] | None = None,
        inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
        equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
        generator: Callable[[int, int, int | None], Tensor] | None = None,
        fixed_X_fantasies: Tensor | None = None,
    ) -> Tensor:
        if inequality_constraints is not None:
            raise ValueError(
                "Inequality constraints are not supported in random initial condition generator"
            )
        if equality_constraints is not None:
            raise ValueError(
                "Equality constraints are not supported in random initial condition generator"
            )
        if generator is not None:
            raise ValueError(
                "Custom generator is not supported in random initial condition generator"
            )
        if fixed_X_fantasies is not None:
            raise ValueError(
                "Fixed X fantasies are not supported in random initial condition generator"
            )

        # generate random points within the bounds
        print("getting random initial conditions")
        start = time.time()
        lower, upper = bounds[0], bounds[1]
        rand = torch.rand(
            10 ** lower.shape[0],
            q,
            lower.shape[0],
            dtype=lower.dtype,
            device=lower.device,
        )
        X = lower + (upper - lower) * rand

        # Apply fixed features if provided
        if fixed_features is not None:
            for idx, val in fixed_features.items():
                X[..., idx] = val

        # Apply nonlinear constraints
        mask = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
        for constraint in nonlinear_constraints:
            mask &= nonlinear_constraint_is_feasible(constraint[0], constraint[1], X)
        X = X[mask]

        # If not enough points, resample until enough
        n_resamples = 0
        while X.shape[0] < num_restarts and n_resamples < max_resamples:
            # print(f"Resampling: {X.shape[0]} < {num_restarts}")
            rand = torch.rand(
                10 ** lower.shape[0],
                q,
                lower.shape[0],
                dtype=lower.dtype,
                device=lower.device,
            )
            new_X = lower + (upper - lower) * rand
            if fixed_features is not None:
                for idx, val in fixed_features.items():
                    new_X[:, idx] = val

            mask = torch.ones(new_X.shape[0], dtype=torch.bool, device=new_X.device)
            for constraint in nonlinear_constraints:
                mask &= nonlinear_constraint_is_feasible(
                    constraint[0], constraint[1], new_X
                )
            new_X = new_X[mask]
            X = torch.cat([X, new_X], dim=0)
            n_resamples += 1

        # If still not enough points, raise an error
        if X.shape[0] < num_restarts:
            raise ValueError("No valid initial conditions found")

        print(
            f"Generated {X.shape[0]} random valid initial conditions (using {num_restarts} of them) over {n_resamples} resamples, took {time.time() - start:.2f} seconds"
        )
        return X[:num_restarts]

    return random_ic_generator
