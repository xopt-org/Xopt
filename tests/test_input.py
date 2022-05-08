import pytest

from xopt.io import read_config_dict, get_function
from xopt.base import Xopt
import yaml

YAML = """
xopt:
    asynch: True
    timeout: 1.0
generator:
    name: MOBO
    optim: 
        num_restarts: 100
evaluator:
    function: xopt.resources.test_functions.tnk.evaluate_TNK

vocs:
    variables:
        x1: [0, 3.14159]
        x2: [0, 3.14159]
    objectives: {y1: MINIMIZE, y2: MINIMIZE}
    constraints:
        c1: [GREATER_THAN, 0]
        c2: ['LESS_THAN', 0.5]
"""


class TestXoptInput:
    def test_input_from_dict(self):
        config = yaml.safe_load(YAML)

        outputs = dict(
            zip(("generator", "evaluator", "vocs", "options"), read_config_dict(config))
        )
        X = Xopt(**outputs)

        assert X.evaluator.function == \
               get_function("xopt.resources.test_functions.tnk.evaluate_TNK")
        assert X.generator.options.optim.num_restarts == 100
        assert X.options.asynch is True

        # test bad options
        config["generator"]["random"] = "BAD"
        with pytest.raises(ValueError):
            read_config_dict(config)

    def test_input_from_yaml(self):
        # dump yaml to file
        with open("test.yaml", "w") as f:
            f.write(YAML)

        X = Xopt.from_yaml_file("test.yaml")

        assert X.evaluator.function == \
               get_function("xopt.resources.test_functions.tnk.evaluate_TNK")
        assert X.generator.options.optim.num_restarts == 100
        assert X.options.asynch is True

        # clean up
        import os
        os.remove("test.yaml")