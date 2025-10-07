import pandas as pd
import pytest

# https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html
# Note that this majorly changes the behavior of pandas - we need to prepare for 3.0
# For example, to_numpy() will return a read-only view of the data unless copy is requested
# Torch will then complain that: The given NumPy array is not writable, and PyTorch does not support non-writable
# tensors. This means writing to this tensor will result in undefined behavior.
# Solution is to either request a copy or mark array writable if dataframe is to be discarded
pd.options.mode.chained_assignment = "raise"
pd.options.mode.copy_on_write = True


def pytest_addoption(parser):
    parser.addoption(
        "--run_compile_tests",
        action="store_true",
        default=False,
        help="run compile tests",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "compilation_test: mark test as compilation_test"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_compile_tests"):
        return
    skip_slow = pytest.mark.skip(reason="compile tests not requested")
    for item in items:
        if "compilation_test" in item.keywords:
            item.add_marker(skip_slow)
