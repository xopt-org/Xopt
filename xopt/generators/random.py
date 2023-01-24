import pandas as pd

from xopt.generator import Generator, GeneratorOptions


class RandomGenerator(Generator):
    alias = "random"

    def __init__(self, vocs, options: GeneratorOptions = None):
        options = options or GeneratorOptions()
        if not isinstance(options, GeneratorOptions):
            raise ValueError("options must be of type GeneratorOptions")
        super().__init__(vocs, options)

    @staticmethod
    def default_options() -> GeneratorOptions:
        return GeneratorOptions()

    def generate(self, n_candidates) -> pd.DataFrame:
        """generate uniform random data points"""
        return pd.DataFrame(self.vocs.random_inputs(n_candidates))
