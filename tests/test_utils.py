from xopt.resources.testing import TEST_VOCS_DATA, TEST_VOCS_BASE
from xopt.utils import add_constraint_information


class TestUtils:
    def test_get_constraint_info(self):
        out = add_constraint_information(TEST_VOCS_DATA, TEST_VOCS_BASE)
