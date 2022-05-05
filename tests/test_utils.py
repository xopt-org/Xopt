from xopt.resources.testing import TEST_VOCS_DATA, TEST_VOCS_BASE
from xopt.utils import add_constraint_information, get_generator_and_defaults
from xopt.generators import registry


class TestUtils:
    def test_get_constraint_info(self):
        out = add_constraint_information(TEST_VOCS_DATA, TEST_VOCS_BASE)

    def test_get_generators(self):
        for name in registry.keys():
            get_generator_and_defaults(name)
