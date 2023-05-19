from typing import List

from pydantic import Field

from xopt.pydantic import XoptBaseModel


class AcquisitionOptions(XoptBaseModel):
    """Options for defining the acquisition function in BO"""

    # monte carlo options
    monte_carlo_samples = Field(128, description="number of monte carlo samples to use")

    proximal_lengthscales: List[float] = Field(
        None, description="lengthscales for proximal biasing"
    )
    use_transformed_proximal_weights: bool = Field(
        True, description="use transformed proximal weights"
    )


class OptimizationOptions(XoptBaseModel):
    """Options for optimizing the acquisition function in BO"""

    num_restarts: int = Field(
        20, description="number of restarts during acquistion function optimization"
    )
    raw_samples: int = Field(
        20, description="number of raw samples used to seed optimization"
    )
    sequential: bool = Field(
        True,
        description="flag to use sequential optimization for q-batch point "
        "selection",
    )
    max_travel_distances: List[float] = Field(
        None,
        description="limits for travel distance between points in normalized space",
    )
    use_turbo: bool = Field(
        False,
        description="flag to use Trust region Bayesian Optimization (TuRBO) "
        "for local optimization",
    )
    first_call: bool = Field(False, exclude=True)
