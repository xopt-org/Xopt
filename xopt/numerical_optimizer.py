import warnings
from abc import ABC, abstractmethod
from typing import Optional

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from pydantic import ConfigDict, Field, PositiveFloat, PositiveInt, field_validator
from torch import Tensor

from xopt.pydantic import XoptBaseModel


class NumericalOptimizer(XoptBaseModel, ABC):
    """
    Base class for numerical optimizers.

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
        Number of restarts (independent initial conditions) for optimization, default is 20.
    n_raw_samples : PositiveInt
        Number of raw samples used to pick the initial `n_restarts` points, default is 128.
    max_iter : PositiveInt
        Maximum number of iterations for the optimizer, default is 2000.
    max_time : Optional[PositiveFloat]
        Maximum time allowed for optimization, default is None (no time limit).
    max_ls : Optional[PositiveInt]
        Maximum number of line search steps, default is None (use scipy defaults).
    with_grad : bool
        Whether to use autograd (True, default) or finite difference (False) for gradient computation.
    ftol: Optional[PositiveFloat]
        Convergence tolerance on the acquisition function value. If None, use scipy defaults.
    pgtol: Optional[PositiveFloat]
        Convergence tolerance on the projected gradient. If None, use scipy defaults.
    sequential : bool
        Use sequential (True) or joint (False, default) optimization when multiple candidates are requested.

    Methods
    -------
    optimize(function, bounds, n_candidates=1, **kwargs)
        Optimize the given acquisition function within the specified bounds.
    """

    name: str = Field("LBFGS", frozen=True)
    n_restarts: PositiveInt = Field(
        20,
        description="number of restarts (independent initial conditions) for optimization",
    )
    n_raw_samples: Optional[PositiveInt] = Field(
        128,
        description="number of raw samples - `n_restarts` best ones are selected from this set",
    )
    max_iter: PositiveInt = Field(
        2000, description="maximum number of optimization steps"
    )
    max_time: Optional[PositiveFloat] = Field(
        5.0, description="maximum time for optimization in seconds"
    )
    max_ls: Optional[PositiveInt] = Field(
        None,
        description="maximum number of line search steps. If None, use scipy defaults.",
    )
    with_grad: bool = Field(
        True,
        description="Use autograd (true) or finite difference (false) for gradient computation."
        " Use autograd unless it is impossible (e.g. non-differentiable elements).",
    )
    eps: Optional[PositiveFloat] = Field(
        None,
        description="Step size used for finite difference if with_grad is False."
        " If None, use scipy default of 1e-08.",
    )
    ftol: Optional[PositiveFloat] = Field(
        None,
        description="Convergence tolerance on the acquisition function value."
        " If None, use scipy default of ftol = 1e7 * numpy.finfo(float).eps = 2.22e-09.",
    )
    pgtol: Optional[PositiveFloat] = Field(
        None,
        description="Convergence tolerance on the projected gradient."
        " If None, use scipy default of 1e-05.",
    )
    sequential: bool = Field(
        False,
        description="Use sequential (true) or joint (false) optimization when q > 1."
        " In practice, joint optimization is faster but requires more GPU memory, "
        " and sequential optimization is slightly more robust (especially with high q)",
    )
    model_config = ConfigDict(validate_assignment=True)

    @field_validator("n_raw_samples", mode="after")
    def validate_n_raw_samples(cls, v, info):
        if v is None:
            return 128
        n_restarts = info.data.get("n_restarts", 20)
        if v < n_restarts:
            warnings.warn(
                f"n_raw_samples should be >= than n_restarts, setting to n_restarts={n_restarts}"
            )
            v = n_restarts
        return v

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

        # empirical testing showed that the max time is overrun slightly on the botorch side
        # fix by slightly reducing the max time passed to this function
        if self.max_time is not None:
            max_time = self.max_time * 0.8 - 0.01
        else:
            max_time = None

        options = {
            "maxiter": self.max_iter,
            "with_grad": self.with_grad,
        }
        if self.ftol is not None:
            options["ftol"] = self.ftol
        if self.pgtol is not None:
            options["pgtol"] = self.pgtol
        if self.max_ls is not None:
            options["max_ls"] = self.max_ls

        candidates, _ = optimize_acqf(
            acq_function=function,
            bounds=bounds,
            q=n_candidates,
            raw_samples=self.n_raw_samples,
            num_restarts=self.n_restarts,
            timeout_sec=max_time,
            options=options,
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

    def optimize(self, function, bounds, n_candidates=1):
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
        mesh_pts = torch.stack(xx).flatten(start_dim=1).T.double()

        # evaluate the function on grid points
        f_values = function(mesh_pts.unsqueeze(1))

        # get the best n_candidates
        _, indicies = torch.sort(f_values)
        x_min = mesh_pts[indicies.squeeze().flipud()]
        return x_min[:n_candidates]
