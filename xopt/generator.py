import pandas as pd
from abc import ABC, abstractmethod
from typing import List


class Generator(ABC):
    _is_done = False

    def __init__(self, vocs):
        self._vocs = vocs

    @abstractmethod
    def generate(self, data: pd.DataFrame, n_candidates) -> List:
        """
        generate `n_candidates` candidates

        """
        pass

    @property
    def is_done(self):
        return self._is_done
