from typing import Callable, List

from pydantic import BaseModel, create_model, Field, root_validator

from xopt.generator import GeneratorOptions
from xopt.generators.bayesian.models.standard import create_standard_model
from xopt.pydantic import JSON_ENCODERS, XoptBaseModel
from xopt.utils import get_function, get_function_defaults


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
        True, description="flag to use local samples to start acqf optimization"
    )
    max_travel_distances: List[float] = Field(
        None,
        description="limits for travel distance between points in normalized space",
    )


class ModelOptions(BaseModel):
    """Options for defining the GP model in BO"""

    function: Callable
    kwargs: BaseModel

    class Config:
        arbitrary_types_allowed = True
        json_encoders = JSON_ENCODERS
        extra = "forbid"
        allow_mutation = False

    @root_validator(pre=True)
    def validate_all(cls, values):
        if "function" in values.keys():
            f = get_function(values["function"])
        else:
            f = create_standard_model

        kwargs = values.get("kwargs", {})
        kwargs = {**get_function_defaults(f), **kwargs}
        values["function"] = f
        values["kwargs"] = create_model("kwargs", **kwargs)()

        return values


class BayesianOptions(GeneratorOptions):
    optim: OptimOptions = OptimOptions()
    acq: AcqOptions = AcqOptions()
    model: ModelOptions = ModelOptions()

    n_initial: int = Field(
        3, description="number of random initial points to measure during first step"
    )


if __name__ == "__main__":
    options = BayesianOptions()
    options.optim.raw_samples = 30
    print(options.dict())

    print(BayesianOptions.schema())
