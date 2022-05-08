import pandas as pd

from xopt.generator import Generator, GeneratorOptions


class RandomOptions(GeneratorOptions):
    class Config:
        title = "random"


class RandomGenerator(Generator):
    def __init__(self, vocs, options: GeneratorOptions = RandomOptions()):
        super(RandomGenerator, self).__init__(vocs, options)

    def generate(self, n_candidates) -> pd.DataFrame:
        """generate uniform random data points"""
        return pd.DataFrame(self.vocs.random_inputs(n_candidates))
