import yaml

from xopt import Evaluator, Xopt
from xopt.generators import RandomGenerator
from xopt.resources.testing import TEST_VOCS_BASE
from xopt.io import state_to_dict, read_config_dict, load_state_yaml
from xopt.evaluator import EvaluatorOptions


def dummy():
    pass


class Test_IO:
    def test_state_to_dict(self):
        evaluator = Evaluator.from_options(EvaluatorOptions(function=dummy))
        generator = RandomGenerator(TEST_VOCS_BASE)

        X = Xopt(generator=generator, evaluator=evaluator, vocs=TEST_VOCS_BASE)
        state_dict = state_to_dict(X)
        assert state_dict["generator"]["name"] == generator.options.__config__.title

        # read from dict
        gen, ev, vcs, options = read_config_dict(state_dict)
        assert ev.options.function == dummy

    def test_load_state_yaml(self):
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
        out = load_state_yaml(yaml.safe_load(YAML))



