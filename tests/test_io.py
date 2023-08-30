from copy import copy

import yaml

from xopt.base import parse_config, state_to_dict, Xopt
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
        print(X.options.model_dump_json())
        print(X.options.serialize_json_str_custom(base_key='bk'))

    def test_state_to_dict(self):
        evaluator = Evaluator(function=dummy)
        generator = RandomGenerator(vocs=TEST_VOCS_BASE)

        X = Xopt(generator=generator, evaluator=evaluator, vocs=TEST_VOCS_BASE)
        state_dict = state_to_dict(X)
        assert state_dict["generator"]["name"] == generator.name
        print(state_dict)

        # read from dict
        config = parse_config(state_dict)
        assert config["evaluator"].function == dummy

    def test_parse_config(self):
        parse_config(yaml.safe_load(copy(TEST_YAML)))
