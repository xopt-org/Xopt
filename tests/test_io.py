from copy import copy

import yaml

from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.random import RandomGenerator
from xopt.resources.testing import TEST_VOCS_BASE, TEST_YAML


def dummy():
    pass


class Test_IO:
    def test_state_to_dict(self):
        evaluator = Evaluator(function=dummy)
        generator = RandomGenerator(vocs=TEST_VOCS_BASE)

        X = Xopt(generator=generator, evaluator=evaluator, vocs=TEST_VOCS_BASE)
        state_dict = X.dict()
        assert state_dict["generator"]["name"] == generator.name

        # load from dict
        X.parse_obj(state_dict)

    def test_parse_config(self):
        Xopt.from_yaml(copy(TEST_YAML))
