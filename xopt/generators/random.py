from xopt.generator import Generator
from xopt.vocs import random_inputs


class RandomGenerator(Generator):
    """
    Random number generator.
    """

    name = "random"
    supports_batch_generation: bool = True
    supports_multi_objective: bool = True
    supports_single_objective: bool = True
    supports_constraints: bool = True

    def generate(self, n_candidates) -> list[dict]:
        """generate uniform random data points"""
        return random_inputs(self.vocs, n_candidates, include_constants=False)
