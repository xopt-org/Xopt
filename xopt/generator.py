import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional

import pandas as pd
from pydantic import ConfigDict, Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from xopt.pydantic import XoptBaseModel
from xopt.vocs import VOCS

logger = logging.getLogger(__name__)


class Generator(XoptBaseModel, ABC):
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

    vocs: VOCS = Field(description="generator VOCS", exclude=True)
    data: Optional[pd.DataFrame] = Field(
        None, description="generator data", exclude=True
    )

    model_config = ConfigDict(validate_assignment=True)

    _is_done = False

    @field_validator("vocs", mode="after")
    def validate_vocs(cls, v, info: ValidationInfo):
        if v.n_objectives > 1 and not info.data["supports_multi_objective"]:
            raise ValueError(
                "this generator only supports a single objective " "specified in vocs"
            )
        return v

    @field_validator("data", mode="before")
    def validate_data(cls, v):
        if isinstance(v, dict):
            try:
                v = pd.DataFrame(v)
            except IndexError:
                v = pd.DataFrame(v, index=[0])
        return v

    def __init__(self, **kwargs):
        """
        Initialize the generator.

        """
        super().__init__(**kwargs)
        logger.info(f"Initialized generator {self.name}")

    @property
    def is_done(self):
        return self._is_done

    @abstractmethod
    def generate(self, n_candidates) -> list[dict]:
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

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        """overwrite model dump to remove faux class attrs"""

        res = super().model_dump(*args, **kwargs)

        res.pop("supports_batch_generation", None)
        res.pop("supports_multi_objective", None)

        return res
