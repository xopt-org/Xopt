import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict


class Generator(ABC):
    _is_done = False

    def __init__(self, vocs):
        self._vocs = vocs

    @abstractmethod
    def generate(self, data: pd.DataFrame, n_candidates) -> List[Dict]:
        """
        generate `n_candidates` candidates

        """
        pass

    @property
    def is_done(self):
        return self._is_done

    @property
    def vocs(self):
        return self._vocs

    def convert_numpy_candidates(self, candidates: np.array):
        list_candidates = []
        for candidate in candidates:
            list_candidates += [dict(zip(self.vocs.variables.keys(), candidate))]

        return list_candidates
