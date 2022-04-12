import numpy as np

from xopt.vocs import VOCS
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA


class TestVOCS(object):
    def test_init(self):
        from xopt.vocs import VOCS

        vocs = VOCS()

    def test_append_constraints(self):
        vocs = TEST_VOCS_BASE
        data = TEST_VOCS_DATA.copy()
        vocs.append_constraints(data)

        assert np.array_equal(
            (data[[f"{ele}_f" for ele in vocs.constraint_names]] <= 0)
            .to_numpy()
            .flatten(),
            data["feasibility"].to_numpy(),
        )
