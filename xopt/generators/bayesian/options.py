from typing import List, Type

from pydantic import Field

from xopt.generator import GeneratorOptions
from xopt.generators.bayesian.base_model import ModelConstructor
from xopt.pydantic import get_descriptions_defaults, JSON_ENCODERS, XoptBaseModel


class AcqOptions(XoptBaseModel):
    """Options for defining the acquisition function in BO"""

    # monte carlo options
    monte_carlo_samples = Field(128, description="number of monte carlo samples to use")

    proximal_lengthscales: List[float] = Field(
        None, description="lengthscales for proximal biasing"
    )
    use_transformed_proximal_weights: bool = Field(
        True, description="use transformed proximal weights"
    )


class OptimOptions(XoptBaseModel):
    """Options for optimizing the acquisition function in BO"""

    num_restarts: int = Field(
        5, description="number of restarts during acquistion " "function optimization"
    )
    raw_samples: int = Field(
        20, description="number of raw samples used to seed optimization"
    )
    sequential: bool = Field(
        True,
        description="flag to use sequential optimization for q-batch point "
        "selection",
    )
    use_nearby_initial_points: bool = Field(
        False, description="flag to use local samples to start acqf optimization"
    )
    max_travel_distances: List[float] = Field(
        None,
        description="limits for travel distance between points in normalized space",
    )


class ModelOptions(XoptBaseModel):
    """Options for defining the GP model in BO"""

    name: str = Field("standard", description="name of model constructor")
    custom_constructor: Type[ModelConstructor] = Field(
        None,
        description="custom model constructor definition, overrides specified name",
    )
    use_low_noise_prior: bool = Field(
        True, description="specify if model should " "assume a low noise environmen"
    )


class BayesianOptions(GeneratorOptions):
    optim: OptimOptions = OptimOptions()
    acq: AcqOptions = AcqOptions()
    model: ModelOptions = ModelOptions()

    n_initial: int = Field(
        3, description="number of random initial points to measure during first step"
    )
    use_cuda: bool = Field(
        False, description="use cuda (GPU) to do bayesian optimization"
    )

    class Config:
        arbitrary_types_allowed = True
        json_encoders = JSON_ENCODERS
        extra = "forbid"


if __name__ == "__main__":
    options = BayesianOptions()
    options.optim.raw_samples = 30

    print(get_descriptions_defaults(options))
