import logging
from abc import ABC, abstractmethod
from typing import Type, TypeVar

import pandas as pd

from xopt.pydantic import XoptBaseModel
from xopt.vocs import VOCS

logger = logging.getLogger(__name__)


class GeneratorOptions(XoptBaseModel):
    """
    Options for the generator.
    """

    pass


_GeneratorOptions = TypeVar("_GeneratorOptions", bound=GeneratorOptions)


class Generator(ABC):
    alias = None

    def __init__(
        self, vocs: VOCS, options: Type[_GeneratorOptions] = GeneratorOptions()
    ):
        """
        Initialize the generator.

        Args:
            vocs: The vocs to use.
            options: The options to use.
        """
        logger.info(f"Initializing generator {self.alias},")

        if not isinstance(options, GeneratorOptions):
            raise TypeError("options must be of type GeneratorOptions")

        self._vocs = vocs.copy()
        self._options = options.copy()
        self._is_done = False
        self._data = pd.DataFrame()
        self._check_options(self._options)

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

    def _check_options(self, options: XoptBaseModel):
        """
        Raise error if options are not compatable, overwrite in each generator if needed
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

    @property
    def options(self):
        return self._options
