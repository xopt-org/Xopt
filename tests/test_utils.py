import numpy as np
import pandas as pd
import pytest

from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA
from xopt.utils import add_constraint_information, explode_all_columns


class TestUtils:
    def test_get_constraint_info(self):
        add_constraint_information(TEST_VOCS_DATA, TEST_VOCS_BASE)

    def test_explode_all_columns(self):
        data = pd.DataFrame(
            {
                "a": [0, 1, 2],
                "b": [np.random.rand(2), np.random.rand(2), np.random.rand(2)],
                "c": [[1, 5], [-7, 8], [100, 122]],
            }
        )

        exploded_data = explode_all_columns(data)
        assert len(exploded_data) == 6

        # pass a bad dataframe
        data = pd.DataFrame(
            {
                "a": [0, 1, 2],
                "b": [np.random.rand(2), np.random.rand(1), np.random.rand(2)],
                "c": [[1, 5], [-7, 8], [100]],
            }
        )
        with pytest.raises(ValueError):
            explode_all_columns(data)
