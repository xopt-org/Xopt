from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np

from xopt.vocs import VOCS


class Problem(ABC):
    name: str = None
    _bounds: list = None
    _start = None
    _optimal_value = None
    _supports_constaints = False

    def __init__(self, n_var, n_obj, n_constr=0) -> None:
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_constr = n_constr
        self.constraint = n_constr > 0

    @property
    def VOCS(self):
        vocs_dict = {
            "variables": {f"x{i + 1}": self._bounds[i] for i in range(self.n_var)},
            "objectives": {f"y{i + 1}": "MAXIMIZE" for i in range(self.n_obj)},
        }
        return VOCS(**vocs_dict)

    @property
    def bounds(self):
        return self._bounds

    @property
    def bounds_numpy(self):
        # 2 x d
        return np.vstack(self._bounds).T

    @abstractmethod
    def _evaluate(self, x: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def evaluate(self, x: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        assert x.shape[-1] == self.n_var, f"{x.shape[-1]} != {self.n_var}"
        size = x.shape[-1]
        for i in range(size):
            if np.any(x[..., i] > self._bounds[i][1]):
                raise ValueError(f"Input {x} greater than {self._bounds[i][1]}")
            if np.any(x[..., i] < self._bounds[i][0]):
                raise ValueError(f"Input {x} lower than {self._bounds[i][0]}")
        return self._evaluate(x, **kwargs)

    def evaluate_dict(self, inputs: Dict, **kwargs):
        ind = np.array([inputs[f"x{i + 1}"] for i in range(self.n_var)])
        objectives, constraints = self.evaluate(ind[None, :], **kwargs)
        outputs = {}
        for i in range(self.n_obj):
            outputs[f"y{i + 1}"] = objectives[0, i].item()
        for i in range(self.n_constr):
            outputs[f"c{i + 1}"] = constraints[0, i].item()
        return outputs

    @property
    def optimal_value(self) -> float:
        return self._optimal_value
