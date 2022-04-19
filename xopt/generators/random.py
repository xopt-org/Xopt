from typing import List, Dict

import pandas as pd

from xopt.generators.generator import Generator


class RandomGenerator(Generator):
    def __init__(self, vocs):
        super(RandomGenerator, self).__init__(vocs)

    def generate(self, n_candidates) -> pd.DataFrame:
        """generate uniform random data points"""
        return pd.DataFrame(self.vocs.random_inputs(n_candidates))
