from typing import List

from pydantic import Field, validator

from xopt.pydantic import XoptBaseModel


class OptimizationOptions(XoptBaseModel):
    """Options for optimizing the acquisition function in BO"""

    # note: num_restarts has to be AFTER raw_samples to allow the validator to catch
    # the default value
    raw_samples: int = Field(
        20,
        description="number of raw samples used to seed optimization",
    )
    num_restarts: int = Field(
        20, description="number of restarts during acquistion function optimization"
    )
    sequential: bool = Field(
        True,
        description="flag to use sequential optimization for q-batch point "
        "selection",
    )

    use_turbo: bool = Field(
        False,
        description="flag to use Trust region Bayesian Optimization (TuRBO) "
        "for local optimization",
    )
    first_call: bool = Field(False, exclude=True)



    class Config:
        validate_assignment = True
