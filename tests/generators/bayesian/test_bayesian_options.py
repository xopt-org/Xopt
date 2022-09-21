from xopt.generators.bayesian.models.standard import create_standard_model
from xopt.generators.bayesian.options import AcqOptions, ModelOptions, OptimOptions, \
    BayesianOptions
from xopt.utils import get_function_defaults


class TestBayesianOptions:
    def test_default(self):
        AcqOptions()
        OptimOptions()
        model_options = ModelOptions()
        BayesianOptions()

        assert model_options.model_kwargs == get_function_defaults(
            create_standard_model
        )

        model_options.json()

    def test_json_serialization(self):
        options = BayesianOptions()
        options.json()
