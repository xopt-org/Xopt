import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict


class Generator(ABC):
    _is_done = False
    _data = pd.DataFrame()
    options = {}

    def __init__(self, vocs):
        self._vocs = vocs

    @abstractmethod
    def generate(self, n_candidates) -> pd.DataFrame:
        """
        generate `n_candidates` candidates

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
