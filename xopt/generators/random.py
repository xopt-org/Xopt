from typing import List, Dict

import numpy as np
import pandas as pd

from xopt.generator import Generator


class RandomGenerator(Generator):
    def __init__(self, vocs):
        super(RandomGenerator, self).__init__(vocs)

    def generate(self, data: pd.DataFrame, n_candidates) -> List[Dict]:
        """generate uniform random data points"""
        problem_dim = len(self.vocs.variables)
        random_vals = np.random.rand(n_candidates, problem_dim)

        # scale according to vocs limits
        for idx, item in enumerate(self.vocs.variables.items()):
            random_vals[:, idx] = (
                random_vals[:, idx] * (item[1][1] - item[1][0]) + item[1][0]
            )

        return self.convert_numpy_candidates(random_vals)
