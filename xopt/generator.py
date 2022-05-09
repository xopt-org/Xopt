import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from xopt.vocs import VOCS

from typing import List, Dict, Type

from xopt.pydantic import XoptBaseModel

import logging

logger = logging.getLogger(__name__)


class GeneratorOptions(XoptBaseModel):
    """
    Options for the generator.
    """
    pass


class Generator(ABC):
    alias = None

    def __init__(self, vocs: VOCS, options: GeneratorOptions = GeneratorOptions()):
        """
        Initialize the generator.

        Args:
            vocs: The vocs to use.
            options: The options to use.
        """
        logger.info(
            f"Initializing generator {self.alias},"
        )

        if not isinstance(options, GeneratorOptions):
            raise TypeError("options must be of type GeneratorOptions")

        self._vocs = vocs

        self._is_done = False
        self._data = pd.DataFrame()
        self.options = options

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

    @staticmethod
    @abstractmethod
    def default_options() -> Type[GeneratorOptions]:
        """
        Get the default options for the generator.
        """
        pass

    @property
    def is_done(self):
        return self._is_done

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame):
        self._data = pd.DataFrame(value)

    @property
    def vocs(self):
        return self._vocs
