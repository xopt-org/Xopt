import logging
from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd
from pydantic import Field

from xopt.pydantic import XoptBaseModel
from xopt.vocs import VOCS

logger = logging.getLogger(__name__)


class Generator(XoptBaseModel, ABC):
    name: ClassVar[str] = Field(description="generator name")
    vocs: VOCS = Field(description="generator VOCS", exclude=True)
    data: pd.DataFrame = Field(
        pd.DataFrame(), description="generator data", exclude=True
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

    # @validator("vocs", pre=True)
    # def vocs_validation(cls, v):
    #     # do vocs first
    #     return v

    class Config:
        validate_assignment = True

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

    @abstractmethod
    def generate(self, n_candidates) -> pd.DataFrame:
        """
        generate `n_candidates` candidates

        """
        pass

    def add_data(self, new_data: pd.DataFrame):
        """
        update dataframe with results from new evaluations.

        This is intended for generators that maintain their own data.

        """
        pass


def _check_vocs(vocs, multi_objective_allowed):
    if vocs.n_objectives != 1 and not multi_objective_allowed:
        raise ValueError("vocs must have one objective for optimization")
