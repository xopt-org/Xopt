from xopt.generator import Generator, support_flag
from xopt.vocs import random_inputs


class RandomGenerator(Generator):
    """
    Random number generator.
    """

    name = "random"
    supports_batch_generation: bool = support_flag(True)
    supports_multi_objective: bool = support_flag(True)
    supports_single_objective: bool = support_flag(True)
    supports_constraints: bool = support_flag(True)
    supports_discrete_variables: bool = support_flag(True)

    def generate(self, n_candidates) -> list[dict]:
        """generate uniform random data points"""
        return random_inputs(self.vocs, n_candidates, include_constants=False)
