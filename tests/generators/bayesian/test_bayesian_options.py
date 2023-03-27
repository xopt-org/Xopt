import pytest

from xopt.generators.bayesian.options import (
    AcqOptions,
    BayesianOptions,
    ModelOptions,
    OptimOptions,
)
from xopt.utils import get_function_defaults


class TestBayesianOptions:
    def test_default(self):
        AcqOptions()
        OptimOptions()
        model_options = ModelOptions()
        BayesianOptions()
        model_options.json()

    def test_json_serialization(self):
        options = BayesianOptions()
        options.json()
