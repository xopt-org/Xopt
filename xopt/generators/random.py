from typing import List, Dict

import numpy as np
import pandas as pd

from xopt.generator import Generator


class RandomGenerator(Generator):
    def __init__(self, vocs):
        super(RandomGenerator, self).__init__(vocs)

    def generate(self, n_candidates) -> List[Dict]:
        """generate uniform random data points"""
        return self.vocs.random_inputs(n_candidates)
