import logging
from abc import ABC, abstractmethod
from typing import ClassVar, Optional

import pandas as pd
from pydantic import ConfigDict, Field, field_validator

from xopt.pydantic import XoptBaseModel
from xopt.vocs import VOCS

logger = logging.getLogger(__name__)


class Generator(XoptBaseModel, ABC):
    name: ClassVar[str] = Field(description="generator name")
    vocs: VOCS = Field(description="generator VOCS", exclude=True)
    data: Optional[pd.DataFrame] = Field(
        None, description="generator data", exclude=True
    )
    supports_batch_generation: ClassVar[bool] = Field(
        default=False,
        description="flag that describes if this "
        "generator can generate "
        "batches of points",
    )
    supports_multi_objective: ClassVar[bool] = Field(
        default=False,
        description="flag that describes if this generator can solve multi-objective "
        "problems",
    )

    model_config = ConfigDict(validate_assignment=True)

    _is_done = False

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

        Args:
            vocs: The vocs to use.
            options: The options to use.
        """
        super().__init__(**kwargs)
        _check_vocs(self.vocs, self.supports_multi_objective)
        logger.info(f"Initialized generator {self.name}")

    @property
    def is_done(self):
        return self._is_done

    @abstractmethod
    def generate(self, n_candidates) -> list[dict]:
        """
        generate `n_candidates` candidates

        """
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


def _check_vocs(vocs, multi_objective_allowed):
    if vocs.n_objectives != 1 and not multi_objective_allowed:
        raise ValueError("vocs must have one objective for optimization")
