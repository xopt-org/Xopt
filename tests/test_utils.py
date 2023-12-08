from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import BaseModel, ConfigDict
from torch import nn

from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA
from xopt.utils import (
    add_constraint_information,
    copy_generator,
    explode_all_columns,
    has_device_field,
)


class MockBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    device: torch.device


class MockModule(nn.Module):
    def __init__(self):
        super(MockModule, self).__init__()
        self.param1 = nn.Parameter(torch.randn(5))
        self.param2 = nn.Parameter(torch.randn(5).to("cpu"))
        self.buffer1 = nn.Parameter(torch.randn(5))
        self.buffer2 = nn.Parameter(torch.randn(5).to("cpu"))


class TestUtils(TestCase):
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

    def test_copy_generator(self):
        generator = MockBaseModel(device=torch.device("cuda"))
        generator_copy, list_of_fields_on_gpu = copy_generator(generator)

        # Check if generator_copy is a deepcopy of generator
        assert generator_copy is not generator
        assert isinstance(generator_copy, MockBaseModel)
        assert generator_copy.device.type == "cuda"

        # Check if list_of_fields_on_gpu contains the correct fields
        assert len(list_of_fields_on_gpu) == 1
        assert list_of_fields_on_gpu[0] == "MockBaseModel"

    def test_has_device_field(self):
        module = MockModule()

        # Check if has_device_field returns True for device "cpu" (which is in the
        # module)
        assert has_device_field(module, torch.device("cpu")) is True

        # Check if has_device_field returns False for device "cuda" (which is not in
        # the module)
        assert has_device_field(module, torch.device("cuda")) is False
