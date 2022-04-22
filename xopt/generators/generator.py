import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from .options import GeneratorOptions
from typing import List, Dict


class Generator(ABC):
    def __init__(self, vocs):
        self._vocs = vocs

        self._is_done = False
        self._data = pd.DataFrame()
        self.options = GeneratorOptions()

    @abstractmethod
    def generate(self, n_candidates) -> pd.DataFrame:
        """
        generate `n_candidates` candidates

        """
        pass

    @abstractmethod
    def update_data(self, new_data: pd.DataFrame):
        """
        update dataframe with results from new evaluations.

        This is intended for generators that maintain their own data.

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
