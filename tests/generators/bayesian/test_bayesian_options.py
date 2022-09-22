import pytest

from xopt.generators.bayesian.models.standard import create_standard_model
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

        assert model_options.kwargs == get_function_defaults(create_standard_model)

        model_options.json()

    def test_json_serialization(self):
        options = BayesianOptions()
        options.json()

    def test_assignment(self):
        options = ModelOptions()
        my_options = {"t": 5}
        with pytest.raises(TypeError):
            options.kwargs = my_options
