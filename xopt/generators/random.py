from xopt.generator import Generator


class RandomGenerator(Generator):
    name = "random"
    supports_batch_generation = True
    supports_multi_objective = True

    def generate(self, n_candidates) -> list[dict]:
        """generate uniform random data points"""
        return self.vocs.random_inputs(n_candidates, include_constants=False)
