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

    def evaluate(self, x, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        assert x.shape[-1] == self.n_var, f"{x.shape[-1]} != {self.n_var}"
        size = x.shape[-1]
        for i in range(size):
            if np.any(x[..., i] > self._bounds[i][1]):
                raise ValueError(f"Input {x} greater than {self._bounds[i][1]}")
            if np.any(x[..., i] < self._bounds[i][0]):
                raise ValueError(f"Input {x} lower than {self._bounds[i][0]}")
        return self._evaluate(x, **kwargs)

    def evaluate_dict(self, inputs: Dict, *args, **kwargs):
        ind = np.array([inputs[f"x{i + 1}"] for i in range(self.n_var)])
        objectives, constraints = self.evaluate(ind[None, :], **kwargs)
        outputs = {"y1": objectives[0].item()}
        if self.constraint:
            outputs["c1"] = constraints[0].item()
        return outputs

    @property
    def optimal_value(self) -> float:
        return self._optimal_value


class MOProblem(Problem):
    @property
    @abstractmethod
    def ref_point(self) -> np.ndarray:
        pass

    @property
    def ref_point_dict(self):
        rp = self.ref_point
        return {f"y{i + 1}": rp[i] for i in range(self.n_obj)}

    def evaluate_dict(self, inputs: Dict, *args, **params):
        ind = np.array([inputs[f"x{i + 1}"] for i in range(self.n_var)])
        objectives, constraints = self.evaluate(ind[None, :])
        outputs = {}
        for i in range(self.n_obj):
            outputs[f"y{i + 1}"] = objectives[0, i].item()
        for i in range(self.n_constr):
            outputs[f"c{i + 1}"] = constraints[0, i].item()
        return outputs


class QuadraticMO(MOProblem):
    _ref_point = np.array([5.0, 5.0])

    def __init__(self, n_var=3, scale=1.0, offset=2.5, negate=False):
        # negate -> maximization
        super().__init__(n_var, n_obj=2)
        self.scale = scale
        self.offset = offset
        self.negate = negate
        self._bounds = [(0, 6.0) for _ in range(n_var)]
        self.vocs = self.VOCS
        self.shift = 0

    @property
    def ref_point(self):
        rp = self._ref_point
        rp = rp**self.n_var
        if self.shift:
            rp += self.shift
        print(f"Quad ref point: {rp}")
        return rp

    @property
    def VOCS(self):
        op = "MAXIMIZE" if self.negate else "MINIMIZE"
        vocs_dict = {
            "objectives": {"y1": op, "y2": op},
            "variables": {f"x{i + 1}": self._bounds[i] for i in range(self.n_var)},
        }
        return VOCS(**vocs_dict)

    def _evaluate(self, x, *args, **kwargs):
        # Keep objectives roughly at unit magnitude
        assert x.shape[-1] == self.n_var
        dim_factor = (1.0**2) ** self.n_var
        scale = self.scale / dim_factor
        objective1 = (
            scale * np.linalg.norm(x - self.offset, axis=-1, keepdims=True) ** 2
        )
        objective2 = scale * np.linalg.norm(x, axis=-1, keepdims=True) ** 2
        objective = np.hstack([objective1.reshape(-1, 1), objective2.reshape(-1, 1)])
        objective = objective if not self.negate else -objective
        return objective, None


class LinearMO(MOProblem):
    def __init__(self, n_var=8, scale=1.0, offset=2.5, negate=False):
        # negate -> maximization
        super().__init__(n_var, n_obj=2)
        self.scale = scale
        self.offset = offset
        self.negate = negate
        self._bounds = [(0, 6.0) for _ in range(n_var)]
        self.vocs = self.VOCS
        self.shift = 0.0

    @property
    def ref_point(self):
        rp = 5.0
        rp = np.sqrt(self.n_var * rp**2)
        if self.shift:
            rp += self.shift
        rp_array = np.array([rp, rp])
        print(f"Linear ref point: {rp_array}")
        return rp_array

    @property
    def VOCS(self):
        op = "MAXIMIZE" if self.negate else "MINIMIZE"
        vocs_dict = {
            "objectives": {"y1": op, "y2": op},
            "variables": {f"x{i + 1}": self._bounds[i] for i in range(self.n_var)},
        }
        return VOCS(**vocs_dict)

    def _evaluate(self, x, *args, **kwargs):
        assert x.shape[-1] == self.n_var
        objective1 = self.scale * np.linalg.norm(
            x - self.offset, axis=-1, keepdims=True
        )
        objective2 = self.scale * np.linalg.norm(x, axis=-1, keepdims=True)
        objective = np.hstack([objective1.reshape(-1, 1), objective2.reshape(-1, 1)])
        objective = objective if not self.negate else -objective
        return objective, None
