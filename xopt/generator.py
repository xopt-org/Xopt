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

    @data.setter
    def data(self, value: pd.DataFrame):
        assert isinstance(value, pd.DataFrame)
        self._data = value

    @property
    def vocs(self):
        return self._vocs

    def convert_numpy_to_inputs(self, inputs: np.ndarray) -> List[Dict]:
        """
        convert 2D numpy array to list of dicts (inputs) for evaluation
        Assumes that the columns of the array match correspond to
        `sorted(self.vocs.variables.keys())

        """
        df = pd.DataFrame(inputs, columns=sorted(self.vocs.variables.keys()))
        return self.convert_dataframe_to_inputs(df)

    def convert_dataframe_to_inputs(self, inputs: pd.DataFrame) -> List[Dict]:
        """
        Convert a dataframe candidate locations to a
        list of dicts to pass to executors.
        """
        # make sure that the df keys contain the vocs variables
        if not set(self.vocs.variables.keys()).issubset(set(inputs.keys())):
            raise RuntimeError(f"input dataframe must at least contain the vocs "
                               f"variables")

        in_copy = inputs.copy()

        # append constants
        constants = self.vocs.constants
        if constants is not None:
            for name, val in constants.items():
                in_copy[name] = val

        return in_copy.to_dict("records")

    def get_training_data(self, data: pd.DataFrame = None):
        """
        get training data from dataframe (usually supplied by xopt base)

        """
        if data is None:
            data = self.data

        objective_names = list(sorted(self.vocs.objectives.keys()))
        if self.vocs.constraints is not None:
            constraint_names = list(sorted(self.vocs.constraints.keys()))
        else:
            constraint_names = []

        inputs = data[sorted(self.vocs.variables.keys())].to_numpy()
        outputs = data[objective_names + constraint_names].to_numpy()

        return inputs, outputs
