from unittest import TestCase
from unittest.mock import patch
import datetime
import time
import math
import numpy as np
import os
import pandas as pd
import pytest
import torch
import tempfile
from pydantic import BaseModel, ConfigDict
from torch import nn

from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA
import xopt.utils
from xopt.utils import (
    add_constraint_information,
    copy_generator,
    explode_all_columns,
    get_local_region,
    has_device_field,
    read_csv,
    nsga2_to_cnsga_file_format,
    read_xopt_csv,
    _explode_pandas_modified,
    safe_call,
    get_n_required_fuction_arguments,
    isotime,
    get_function,
    get_function_defaults,
)


# Module-level function for get_function test


def foo():
    return 42


globals()["bar"] = foo
globals()["not_callable"] = 5


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

    def test_get_local_region(self):
        class DummyVOCS:
            variable_names = ["x", "y"]
            variables = {"x": (0.0, 10.0), "y": (1.0, 5.0)}

        vocs = DummyVOCS()
        center_point = {"x": 5.0, "y": 3.0}
        # Normal case
        bounds = get_local_region(center_point, vocs, fraction=0.2)
        assert set(bounds.keys()) == set(["x", "y"])
        # Check bounds are within the variable limits
        assert bounds["x"][0] >= vocs.variables["x"][0]
        assert bounds["x"][1] <= vocs.variables["x"][1]
        assert bounds["y"][0] >= vocs.variables["y"][0]
        assert bounds["y"][1] <= vocs.variables["y"][1]
        # Edge case: center at lower bound
        center_point = {"x": 0.0, "y": 1.0}
        bounds = get_local_region(center_point, vocs, fraction=0.5)
        assert bounds["x"][0] == vocs.variables["x"][0]
        assert bounds["y"][0] == vocs.variables["y"][0]
        # Edge case: center at upper bound
        center_point = {"x": 10.0, "y": 5.0}
        bounds = get_local_region(center_point, vocs, fraction=0.5)
        assert bounds["x"][1] == vocs.variables["x"][1]
        assert bounds["y"][1] == vocs.variables["y"][1]
        # Error case: wrong keys
        with pytest.raises(KeyError):
            get_local_region({"x": 1.0}, vocs)

    def test_read_csv_all_and_last_n_lines(self):
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            fname = f.name
            f.write("col1,col2\n")
            for i in range(10):
                f.write(f"{i},{i * 2}\n")
        try:
            # Read all lines
            df_all = read_csv(fname)
            assert isinstance(df_all, pd.DataFrame)
            assert len(df_all) == 10
            assert list(df_all.columns) == ["col1", "col2"]
            # Read last 3 lines
            df_last3 = read_csv(fname, last_n_lines=3)
            assert len(df_last3) == 3
            assert df_last3.iloc[0, 0] == 7
            assert df_last3.iloc[-1, 1] == 18
            # Read last n lines greater than total (should return all)
            df_all2 = read_csv(fname, last_n_lines=20)
            assert len(df_all2) == 10
            # Test with additional kwargs
            df = read_csv(fname, last_n_lines=2, dtype={"col1": int, "col2": int})
            assert df.dtypes["col1"] is int
        finally:
            os.remove(fname)

    def test__explode_pandas_modified(self):
        # Single row DataFrame, columns to explode
        df = pd.DataFrame(
            {
                "a": [[1, 2, 3]],
                "b": [[4, 5, 6]],
                "c": [10],
            }
        )
        result = _explode_pandas_modified(df, ["a", "b"], 3)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b", "c"]
        assert len(result) == 3
        assert np.all(result["a"] == [1, 2, 3])
        assert np.all(result["b"] == [4, 5, 6])
        assert np.all(result["c"] == [10, 10, 10])

        # Should raise NotImplementedError for multi-row DataFrame
        df2 = pd.DataFrame({"a": [[1, 2]], "b": [[3, 4]]})
        df2 = pd.concat([df2, df2])
        try:
            _explode_pandas_modified(df2, ["a", "b"], 2)
        except NotImplementedError:
            pass
        else:
            assert False, "Expected NotImplementedError for multi-row DataFrame"

    def test_safe_call(self):
        # Test normal function
        def add(a, b):
            time.sleep(0.01)
            return a + b

        result = safe_call(add, 2, 3)
        assert result["result"] == 5
        assert result["exception"] is None
        assert result["traceback"] == ""
        assert result["runtime"] > 0

        # Test function that raises
        def fail():
            raise ValueError("fail!")

        result = safe_call(fail)
        assert result["result"] is None
        assert result["exception"] is not None
        assert "ValueError" in result["traceback"]
        assert result["runtime"] > 0

        # Test function with kwargs
        def kw(a=1):
            return a * 2

        result = safe_call(kw, a=4)
        assert result["result"] == 8
        assert result["exception"] is None
        assert result["traceback"] == ""

    def test_get_n_required_fuction_arguments(self):
        # Function with 2 required, 1 optional
        def f1(a, b, c=1):
            pass

        assert get_n_required_fuction_arguments(f1) == 2

        # Function with all required
        def f2(x, y):
            pass

        assert get_n_required_fuction_arguments(f2) == 2

        # Function with all optional
        def f3(a=1, b=2):
            pass

        assert get_n_required_fuction_arguments(f3) == 0

        # Function with *args and **kwargs
        def f4(a, *args, b=2, **kwargs):
            pass

        assert get_n_required_fuction_arguments(f4) == 1

        # Method (self is not counted as required argument)
        class C:
            def m(self, x, y=2):
                pass

        assert get_n_required_fuction_arguments(C().m) == 1

        def l_func(x, y=5):
            return x

        assert get_n_required_fuction_arguments(l_func) == 1

    def test_isotime(self):
        # Test default (no microseconds)
        tstr = isotime()
        t = datetime.datetime.fromisoformat(tstr)
        assert t.microsecond == 0
        # Test with microseconds
        tstr2 = isotime(include_microseconds=True)
        t2 = datetime.datetime.fromisoformat(tstr2)
        assert isinstance(t2, datetime.datetime)
        # Should be close to now
        now = datetime.datetime.now(t2.tzinfo)
        assert abs((now - t2).total_seconds()) < 10

    def test_get_function(self):
        # Callable input
        assert get_function(foo) is foo
        # Fully qualified name
        f = get_function("math.sqrt")
        assert f is math.sqrt
        # Not a string or callable
        try:
            get_function(123)
        except ValueError:
            pass
        else:
            assert False, "Expected ValueError for non-str/callable input"
        # Not a global or importable function
        try:
            get_function("not_a_function")
        except Exception:
            pass
        else:
            assert False, "Expected Exception for missing function"

    def test_get_function_defaults(self):
        # Function with defaults
        def f(a, b=2, c=3):
            pass

        d = get_function_defaults(f)
        assert d == {"b": 2, "c": 3}

        # Function with no defaults
        def g(x, y):
            pass

        assert get_function_defaults(g) == {}

        # Function with *args, **kwargs, and defaults
        def h(a, b=1, *args, c=2, **kwargs):
            pass

        d2 = get_function_defaults(h)
        assert d2 == {"b": 1}

        def l_func(x, y=5):
            return x

        assert get_function_defaults(l_func) == {"y": 5}


# Module-level test for non-callable global


def test_get_function_non_callable_global():
    with patch.dict(xopt.utils.__dict__, {"not_callable": 5}):
        try:
            get_function("not_callable")
        except ValueError:
            pass
        else:
            assert False, "Expected ValueError for non-callable global"


def test_read_xopt_csv(tmp_path):
    # Create two CSV files with xopt_index as index
    df1 = pd.DataFrame({"xopt_index": [0, 1], "a": [10, 20]})
    df2 = pd.DataFrame({"xopt_index": [2, 3], "a": [30, 40]})
    file1 = tmp_path / "file1.csv"
    file2 = tmp_path / "file2.csv"
    df1.to_csv(file1, index=False)
    df2.to_csv(file2, index=False)
    # Read single file
    result1 = read_xopt_csv(str(file1))
    assert isinstance(result1, pd.DataFrame)
    assert list(result1.index) == [0, 1]
    assert list(result1["a"]) == [10, 20]
    # Read multiple files
    result2 = read_xopt_csv(str(file1), str(file2))
    assert isinstance(result2, pd.DataFrame)
    assert set(result2.index) == {0, 1, 2, 3}
    assert set(result2["a"]) == {10, 20, 30, 40}
    # Check that index is named xopt_index
    assert result2.index.name == "xopt_index"


def test_nsga2_to_cnsga_file_format(tmp_path):
    # Create input_dir and output_dir
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    # Create populations.csv
    pop_data = pd.DataFrame(
        {
            "xopt_generation": [1700000000, 1700000001],
            "a": [1, 2],
            "b": [3, 4],
        }
    )
    pop_data.to_csv(input_dir / "populations.csv", index=False)

    # Create data.csv
    dat_data = pd.DataFrame(
        {
            "xopt_parent_generation": [1700000000, 1700000001],
            "c": [5, 6],
            "d": [7, 8],
        }
    )
    dat_data.to_csv(input_dir / "data.csv", index=False)

    # Run conversion
    nsga2_to_cnsga_file_format(str(input_dir), str(output_dir))

    # Check population files
    for gen in [1700000000, 1700000001]:
        timestamp = (
            datetime.datetime.fromtimestamp(int(gen), tz=datetime.timezone.utc)
            .isoformat()
            .replace(":", "_")
        )
        fname = output_dir / f"cnsga_population_{timestamp}.csv"
        assert fname.exists()
        df = pd.read_csv(fname)
        assert "xopt_generation" in df.columns

    # Check offspring files
    for gen in [1700000000, 1700000001]:
        timestamp = (
            datetime.datetime.fromtimestamp(int(gen) + 1, tz=datetime.timezone.utc)
            .isoformat()
            .replace(":", "_")
        )
        fname = output_dir / f"cnsga_offspring_{timestamp}.csv"
        assert fname.exists()
        df = pd.read_csv(fname)
        assert "xopt_parent_generation" in df.columns
