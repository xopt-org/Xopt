from copy import deepcopy

from xopt.generators.random import RandomGenerator
from xopt.resources.testing import TEST_VOCS_BASE


class TestRandomGenerator:
    def test_random_generator(self):
        gen = RandomGenerator(vocs=TEST_VOCS_BASE)

        # generate samples
        samples = gen.generate(10)
        assert len(samples) == 10

        assert gen.name == "random"

        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives = {"y": "MAXIMIZE", "z": "MINIMIZE"}

        gen = RandomGenerator(vocs=test_vocs)
        gen.yaml()
