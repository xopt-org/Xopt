import logging
from abc import ABC, abstractmethod
from typing import Type, TypeVar, Optional

import pandas as pd
from pydantic import BaseModel

from xopt.pydantic import XoptBaseModel
from xopt.vocs import VOCS

logger = logging.getLogger(__name__)


class GeneratorOptions(XoptBaseModel):
    """
    Options for the generator.
    """

    def partial_update(self, new_kwargs: dict):
        raise NotImplementedError('Generic options do not support updates')


#_GeneratorOptions = TypeVar("_GeneratorOptions", bound=GeneratorOptions)


class Generator(ABC):
    alias = None

    def __init__(
        self, vocs: VOCS, options: GeneratorOptions = None
    ):
        """
        Initialize the generator.

        Args:
            vocs: The vocs to use.
            options: The options to use.
        """
        logger.info(f"Initializing generator {self.alias},")
        options: XoptBaseModel = options or GeneratorOptions()
        if not isinstance(options, GeneratorOptions):
            raise TypeError("options must be of type GeneratorOptions")

        self._vocs = vocs.copy()
        self._options = options.copy()
        self._check_options(self._options)
        self._options_ext = options.copy()
        self._is_done = False
        self._data = pd.DataFrame()


    @abstractmethod
    def generate(self, n_candidates) -> pd.DataFrame:
        """
        generate `n_candidates` candidates

        """
        pass

    def generate_custom(self, n_candidates, config_changes: dict, is_permanent: bool = False):
        """
        Applies an update to generator options, and then
        generates `n_candidates` candidates.

        Args:
            n_candidates: number of candidates to generate
            config_changes: dict with updated settings
            is_permanent: if True, changes are persisted for all future iterations
        """
        # test changes first, this will raise if there are issues
        new_options = self._options_ext.partial_update(config_changes)

        # Shallow copy will be affected by some changes, but deepcopy costly
        previous_options = self._options
        self._options = new_options.copy()
        r = self.generate(n_candidates)
        if not is_permanent:
            self._options = previous_options
            self._options_ext = previous_options.copy()
        return r, new_options

    def add_data(self, new_data: pd.DataFrame):
        """
        update dataframe with results from new evaluations.

        This is intended for generators that maintain their own data.

        """
        pass

    def update_data(self, new_data: pd.DataFrame):
        """
        Change data to the new one, taking care of any internal state.
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
