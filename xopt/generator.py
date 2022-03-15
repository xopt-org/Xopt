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
    def generate(self, n_candidates) -> List[Dict]:
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

    @property
    def vocs(self):
        return self._vocs

    def add_data(self, data: pd.DataFrame):
        self._data = data

    def convert_numpy_candidates(self, candidates: np.array):
        """
        Convert a numpy array of candidate locations to a
        list of dicts to pass to executors. Assumes that the columns of the array
        match the order of variable keys in VOCS
        """
        list_candidates = []
        for candidate in candidates:
            list_candidates += [dict(zip(self.vocs.variables.keys(), candidate))]

        return list_candidates

    def get_bounds(self):
        """
        returns the optimization bounds taken from vocs,
        returns a numpy array of the shape (2, d) where d is the number of input
        parameters
        """
        return np.vstack([np.array(ele) for _, ele in self.vocs.variables.items()]).T

    def get_training_data(self, data: pd.DataFrame = None):
        """
        get training data from dataframe (usually supplied by xopt base)

        """
        if data is None:
            data = self.data

        objective_names = list(self.vocs.objectives.keys())
        if self.vocs.constraints is not None:
            constraint_names = list(self.vocs.constraints.keys())
        else:
            constraint_names = []

        inputs = data[self.vocs.variables.keys()].to_numpy()
        outputs = data[objective_names + constraint_names].to_numpy()

        return inputs, outputs
