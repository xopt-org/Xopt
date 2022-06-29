from xopt.generators import generators
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA
from xopt.utils import add_constraint_information, get_generator_and_defaults


class TestUtils:
    def test_get_constraint_info(self):
        add_constraint_information(TEST_VOCS_DATA, TEST_VOCS_BASE)

    def test_get_generators(self):
        for name in generators.keys():
            get_generator_and_defaults(name)
