import math
from abc import abstractmethod
from typing import Dict

import numpy as np
from scipy.special import gamma

from xopt.resources.test_functions.problem import Problem

from xopt.vocs import VOCS


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
        if ind.ndim == 1:
            # MOBO yields floats
            ind = ind[np.newaxis, :]
        else:
            # Random generator yields length-1 numpy arrays
            ind = ind.T
        objectives, constraints = self.evaluate(ind)
        outputs = {}
        for i in range(self.n_obj):
            outputs[f"y{i + 1}"] = objectives[0, i].item()
        for i in range(self.n_constr):
            outputs[f"c{i + 1}"] = constraints[0, i].item()
        return outputs


class QuadraticMO(MOProblem):
    """Quadratic multi-objective test problem - by default, finding minima with 1 objective offset"""

    _ref_point = np.array([5.0, 5.0])

    def __init__(self, n_var=3, scale=1.0, offset=1.5, negate=False):
        # negate -> maximization
        super().__init__(n_var, n_obj=2)
        self.scale = scale
        self.offset = offset
        self.negate = negate
        self._bounds = [(0, 3.0) for _ in range(n_var)]
        self.vocs = self.VOCS
        self.shift = 0

    @property
    def ref_point(self):
        rp = self._ref_point
        rp = rp**self.n_var
        if self.shift:
            rp += self.shift
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
        # print(f"Linear ref point: {rp_array}")
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


# From BoTorch
class DTLZ2(MOProblem):
    def __init__(self, n_var=3, negate=False):
        # negate -> maximization
        super().__init__(n_var, n_obj=2)
        self.negate = negate
        self._bounds = [(0.0, 1.0) for _ in range(n_var)]
        self.vocs = self.VOCS

    @property
    def VOCS(self):
        op = "MAXIMIZE" if self.negate else "MINIMIZE"
        vocs_dict = {
            "objectives": {"y1": op, "y2": op},
            "variables": {f"x{i + 1}": self._bounds[i] for i in range(self.n_var)},
        }
        return VOCS(**vocs_dict)

    @property
    def ref_point(self):
        return np.ones(self.n_var) * 1.1

    @property
    def _max_hv(self) -> float:
        # hypercube - volume of hypersphere in R^d such that all coordinates are
        # positive
        hypercube_vol = 1.1**self.n_obj
        pos_hypersphere_vol = (
            math.pi ** (self.n_obj / 2) / gamma(self.n_obj / 2 + 1) / 2**self.n_obj
        )
        return hypercube_vol - pos_hypersphere_vol

    def _evaluate(self, X: np.ndarray, **kwargs) -> np.ndarray:
        assert X.shape[1] == self.n_var
        k = X.shape[1] - 2 + 1
        X_m = X[..., -k:]
        g_X = ((X_m - 0.5) ** 2).sum(axis=-1)
        g_X_plus1 = 1 + g_X
        fs = []
        pi_over_2 = np.pi / 2
        for i in range(self.n_obj):
            idx = 2 - 1 - i
            f_i = g_X_plus1.copy()
            f_i *= np.cos(X[..., :idx] * pi_over_2).prod(axis=-1)
            if i > 0:
                f_i *= np.sin(X[..., idx] * pi_over_2)
            fs.append(f_i)
        return np.stack(fs, axis=-1), None
