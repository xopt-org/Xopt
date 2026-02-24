import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Hashable, Optional

import pandas as pd
from pydantic import ConfigDict, Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from xopt.errors import VOCSError
from xopt.pydantic import XoptBaseModel
from xopt.vocs import VOCS

logger = logging.getLogger(__name__)


class Generator(XoptBaseModel, ABC):
    """
    Base class for Generators.

    Generators are responsible for generating new points to evaluate.

    Attributes
    ----------
    name : str
        Name of the generator.
    supports_batch_generation : bool
        Flag that describes if this generator can generate batches of points.
    supports_multi_objective : bool
        Flag that describes if this generator can solve multi-objective problems.
    vocs : VOCS
        Generator VOCS.
    data : pd.DataFrame, optional
        Generator data.
    model_config : ConfigDict
        Model configuration.
    """

    name: ClassVar[str] = Field(description="generator name")

    supports_batch_generation: bool = Field(
        default=False,
        description="flag that describes if this "
        "generator can generate "
        "batches of points",
        frozen=True,
        exclude=True,
    )
    supports_multi_objective: bool = Field(
        default=False,
        description="flag that describes if this generator can solve multi-objective "
        "problems",
        frozen=True,
        exclude=True,
    )
    supports_single_objective: bool = Field(
        default=False,
        description="flag that describes if this generator can solve multi-objective "
        "problems",
        frozen=True,
        exclude=True,
    )
    supports_constraints: bool = Field(
        default=False,
        description="flag that describes if this generator can solve "
        "constrained optimization problems",
        frozen=True,
        exclude=True,
    )

    vocs: VOCS = Field(description="generator VOCS", exclude=True)
    data: Optional[pd.DataFrame] = Field(
        None, description="generator data", exclude=True
    )

    model_config = ConfigDict(validate_assignment=True)

    def __init__(self, **kwargs: Any):
        """
        Initialize the generator.
        """
        super().__init__(**kwargs)
        logger.info(f"Initialized generator {self.name}")

    @field_validator("vocs", mode="after")
    @classmethod
    def validate_vocs(cls, value: VOCS, info: ValidationInfo):
        if value.n_constraints > 0 and not info.data["supports_constraints"]:
            raise VOCSError("this generator does not support constraints")
        if value.n_objectives == 1:
            if not info.data["supports_single_objective"]:
                raise VOCSError(
                    "this generator does not support single objective optimization"
                )
        elif value.n_objectives > 1 and not info.data["supports_multi_objective"]:
            raise VOCSError(
                "this generator does not support multi-objective optimization"
            )

        return value

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, value: Any):
        if isinstance(value, dict):
            try:
                value = pd.DataFrame(value)
            except IndexError:
                value = pd.DataFrame(value, index=[0])
        return value

    @abstractmethod
    def generate(self, n_candidates: int) -> list[dict[Hashable, Any]]:
        pass

    def add_data(self, new_data: pd.DataFrame):
        """
        update dataframe with results from new evaluations.

        This is intended for generators that maintain their own data.

        """
        if self.data is not None:
            self.data = pd.concat([self.data, new_data], axis=0)
        else:
            self.data = new_data

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """overwrite model dump to remove faux class attrs"""

        res = super().model_dump(*args, **kwargs)

        res.pop("supports_batch_generation", None)
        res.pop("supports_multi_objective", None)

        return res


class StateOwner:
    """
    Mix-in class that represents a generator that owns its own state and needs special handling
    of data loading on deserialization.
    """

    def set_data(self, data: pd.DataFrame):
        """
        Set the full dataset for the generator. Typically only used when loading from a save file.

        Parameters
        ----------
        data : pd.DataFrame
            The data to set.
        """
        raise NotImplementedError
