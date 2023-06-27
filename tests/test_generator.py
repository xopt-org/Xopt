from unittest.mock import patch

from xopt.generator import Generator
from xopt.resources.testing import TEST_VOCS_BASE


class TestGenerator:
    @patch.multiple(Generator, __abstractmethods__=set())
    def test_init(self):
        Generator(vocs=TEST_VOCS_BASE)
