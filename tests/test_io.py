from copy import copy

import yaml

from xopt import Evaluator, Xopt
from xopt.generators import RandomGenerator
from xopt.resources.testing import TEST_VOCS_BASE, TEST_YAML
from xopt.io import state_to_dict, parse_config
from xopt.evaluator import EvaluatorOptions


def dummy():
    pass


class Test_IO:
    def test_state_to_dict(self):
        evaluator = Evaluator.from_options(EvaluatorOptions(function=dummy))
        generator = RandomGenerator(TEST_VOCS_BASE)

        X = Xopt(generator=generator, evaluator=evaluator, vocs=TEST_VOCS_BASE)
        state_dict = state_to_dict(X)
        assert state_dict["generator"]["name"] == generator.alias

        # read from dict
        config =  parse_config(state_dict)
        assert config["evaluator"].options.function == dummy

    def test_parse_config(self):
        parse_config(yaml.safe_load(copy(TEST_YAML)))
