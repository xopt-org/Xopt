import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf, optimize_acqf_discrete, optimize_acqf_mixed
from pydantic import Field, PositiveFloat, PositiveInt
from torch import Tensor

from xopt.pydantic import XoptBaseModel


logger = logging.getLogger(__name__)


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
    optimize
        Abstract method to optimize a function to produce a number of candidate points that minimize the function.
    """

    @abstractmethod
    def optimize(
        self,
        function: AcquisitionFunction,
        bounds: Tensor,
        n_candidates: int = 1,
        **kwargs: Any,
    ) -> Tensor:
        """Optimize a function to produce a number of candidate points that minimize the function."""
        raise NotImplementedError


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
    optimize
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
    mixed_max_discrete_configurations: PositiveInt = Field(
        512,
        description=(
            "maximum number of discrete configurations to enumerate in mixed"
            " optimization"
        ),
    )
    discrete_max_choices: PositiveInt = Field(
        4096,
        description="maximum number of discrete choices to evaluate directly",
    )
    mixed_n_restarts: Optional[PositiveInt] = Field(
        None,
        description=(
            "number of restarts for mixed optimization; defaults to n_restarts"
        ),
    )
    mixed_raw_samples: Optional[PositiveInt] = Field(
        None,
        description=(
            "number of raw samples for mixed optimization; defaults to n_restarts"
        ),
    )
    discrete_max_batch_size: PositiveInt = Field(
        2048,
        description="maximum batch size used by botorch discrete optimization",
    )

    def optimize(
        self,
        function: AcquisitionFunction,
        bounds: Tensor,
        n_candidates: int = 1,
        **kwargs: Any,
    ):
        """
        Optimize the given function within the specified bounds using LBFGS.

        Parameters
        ----------
        function : Callable
            The function to be optimized.
        bounds : Tensor
            A tensor specifying the bounds for the optimization. It must have the shape [2, ndim].
        n_candidates : int, optional
            The number of candidates to generate (default is 1).
        **kwargs : dict
            Additional keyword arguments to be passed to the function optimizer.

        Returns
        -------
        candidates : Tensor
            The optimized candidates.
        """

        assert isinstance(bounds, Tensor)
        if len(bounds) != 2:
            raise ValueError("bounds must have the shape [2, ndim]")

        # empirical testing showed that max time is overrun slightly on the botorch side
        # fix by reducing the max time passed to botorch
        if self.max_time is not None:
            max_time = self.max_time * 0.8 - 0.01
        else:
            max_time = None

        fixed_features_list = kwargs.pop("fixed_features_list", None)
        discrete_choices = kwargs.pop("discrete_choices", None)

        if fixed_features_list is not None and discrete_choices is not None:
            raise ValueError(
                "cannot specify both fixed_features_list and discrete_choices"
            )

        if fixed_features_list is not None:
            original_count = len(fixed_features_list)
            fixed_features_list = fixed_features_list[
                : self.mixed_max_discrete_configurations
            ]
            if len(fixed_features_list) == 0:
                raise ValueError("fixed_features_list cannot be empty")

            if original_count > self.mixed_max_discrete_configurations:
                logger.warning(
                    "truncating mixed discrete configurations from %d to %d",
                    original_count,
                    self.mixed_max_discrete_configurations,
                )

            mixed_n_restarts = self.mixed_n_restarts or self.n_restarts
            mixed_raw_samples = self.mixed_raw_samples or self.n_restarts

            candidates, _ = optimize_acqf_mixed(
                acq_function=function,
                bounds=bounds,
                q=n_candidates,
                num_restarts=mixed_n_restarts,
                fixed_features_list=fixed_features_list,
                raw_samples=mixed_raw_samples,
                timeout_sec=max_time,
                options={"maxiter": self.max_iter},
                **kwargs,
            )
            return candidates

        if discrete_choices is not None:
            if "batch_initial_conditions" in kwargs:
                logger.warning(
                    "batch_initial_conditions are not used by optimize_acqf_discrete"
                )
                kwargs.pop("batch_initial_conditions", None)

            if discrete_choices.shape[0] > self.discrete_max_choices:
                logger.warning(
                    "truncating discrete choices from %d to %d",
                    discrete_choices.shape[0],
                    self.discrete_max_choices,
                )
                discrete_choices = discrete_choices[: self.discrete_max_choices]

            candidates, _ = optimize_acqf_discrete(
                acq_function=function,
                q=n_candidates,
                choices=discrete_choices,
                max_batch_size=self.discrete_max_batch_size,
                **kwargs,
            )
            return candidates

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
    optimize
        Optimize the given function within the specified bounds.

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
        Optimize the given function within the specified bounds using a brute-force grid search.

        Parameters
        ----------
        function : Callable
            The function to be optimized.
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
