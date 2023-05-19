import pandas as pd

from xopt.generator import Generator


class RandomGenerator(Generator):
    name = "random"
    supports_batch_generation = True
    supports_multi_objective = True

    def generate(self, n_candidates) -> pd.DataFrame:
        """generate uniform random data points"""
        return pd.DataFrame(self.vocs.random_inputs(n_candidates))
