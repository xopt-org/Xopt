from typing import Callable, List

from pydantic import BaseModel, create_model, Field, root_validator, validate_model, Extra
from pydantic.utils import deep_update

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

    def partial_update(self, new_kwargs: dict) -> 'BayesianOptions':
        """
        Updating model inplace is in general a pain:
        - New values are not auto-validated
        - copy() does not update items recursively
        - Most other operations make unwanted new models

        See:
        https://github.com/pydantic/pydantic/issues/418
        https://github.com/pydantic/pydantic/discussions/3139
        https://github.com/pydantic/pydantic/issues/1864

        This method merges new dictionary into current model,
        replacing values in-place but not making any copies.
        """
        # def biased_merge(a, b, path=None):
        #     if path is None:
        #         path = []
        #     for key in b:
        #         if key in a:
        #             is_a_dict = isinstance(a[key], dict)
        #             is_b_dict = isinstance(b[key], dict)
        #             if is_a_dict and is_b_dict:
        #                 biased_merge(a[key], b[key], path + [str(key)])
        #             elif is_a_dict != is_b_dict:
        #                 # Dicts should be paired since pydantic generates defaults
        #                 raise AttributeError(f'Structure mismatch at {path}:{key}')
        #             else:
        #                 # Leave validation to pydantic
        #                 a[key] = b[key]
        #         else:
        #             a[key] = b[key]
        #     return a
        #updated_dict = biased_merge(self.dict(), new_kwargs)

        updated_dict = deep_update(self.dict(), new_kwargs)
        self.check_and_set(updated_dict)
        return self

    def check_and_set(self, new_dict=None):
        """
        Validates a dictionary against the options model, and if so, sets new values
        """
        # https://github.com/pydantic/pydantic/issues/1864
        new_dict = new_dict or self.__dict__
        values, fields_set, validation_error = validate_model(
            self.__class__, new_dict
        )
        if validation_error:
            raise validation_error
        try:
            object.__setattr__(self, "__dict__", values)
        except TypeError as e:
            raise TypeError(
                "Model values must be a dict; you may not have returned "
                + "a dictionary from a root validator"
            ) from e
        object.__setattr__(self, "__fields_set__", fields_set)


if __name__ == "__main__":
    options = BayesianOptions()
    options.optim.raw_samples = 30
    print(options.dict())

    print(BayesianOptions.schema())
