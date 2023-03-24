import pytest
from pydantic import ValidationError

from xopt.generators.rcds.rcds import RCDSGenerator
from xopt.resources.testing import TEST_VOCS_BASE


class TestRCDSGenerator:
    def test_rcds_generate_multiple_points(self):
        gen = RCDSGenerator(TEST_VOCS_BASE)

        # Try to generate multiple samples
        with pytest.raises(NotImplementedError):
            gen.generate(2)

    def test_rcds_options(self):
        gen = RCDSGenerator(TEST_VOCS_BASE)

        with pytest.raises(ValidationError):
            gen.options.step = 0

        with pytest.raises(ValidationError):
            gen.options.tol = 0
