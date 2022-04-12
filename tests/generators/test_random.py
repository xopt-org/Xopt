from xopt.generators.random import RandomGenerator
from xopt.resources.testing import TEST_VOCS_BASE


class TestRandomGenerator:
    def test_random_generator(self):
        gen = RandomGenerator(TEST_VOCS_BASE)

        # generate samples
        samples = gen.generate(10)
        assert len(samples) == 10
