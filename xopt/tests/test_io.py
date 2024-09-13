from copy import copy

from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.random import RandomGenerator
from xopt.resources.testing import TEST_VOCS_BASE, TEST_YAML


def dummy():
    pass


class Test_IO:
    def test_options_to_dict(self):
        evaluator = Evaluator(function=dummy)
        generator = RandomGenerator(vocs=TEST_VOCS_BASE)
        X = Xopt(generator=generator, evaluator=evaluator, vocs=TEST_VOCS_BASE)
        print(X.model_dump_json())
        print(X.to_json(base_key="bk"))

    def test_state_to_dict(self):
        evaluator = Evaluator(function=dummy)
        generator = RandomGenerator(vocs=TEST_VOCS_BASE)

        X = Xopt(generator=generator, evaluator=evaluator, vocs=TEST_VOCS_BASE)
        state_dict = X.dict()
        assert state_dict["generator"]["name"] == generator.name
        print(state_dict)

        # load from dict
        X.model_validate(state_dict)

    def test_parse_config(self):
        Xopt.from_yaml(copy(TEST_YAML))
