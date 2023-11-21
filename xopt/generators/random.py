from xopt.generator import Generator


class RandomGenerator(Generator):
    """
    Random number generator.
    """

    name = "random"
    supports_batch_generation: bool = True
    supports_multi_objective: bool = True

    def generate(self, n_candidates) -> list[dict]:
        """generate uniform random data points"""
        return self.vocs.random_inputs(n_candidates, include_constants=False)
