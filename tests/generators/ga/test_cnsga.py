import yaml

from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.ga.cnsga import CNSGAGenerator
from xopt.resources.test_functions.tnk import evaluate_TNK, tnk_vocs
from xopt.resources.testing import TEST_YAML


def test_cnsga():
    X = Xopt(
        generator=CNSGAGenerator(vocs=tnk_vocs),
        evaluator=Evaluator(function=evaluate_TNK),
        vocs=tnk_vocs,
        max_evaluations=5
    )
    X.run()


def test_cnsga_from_yaml():
    X = Xopt.parse_obj(yaml.safe_load(TEST_YAML))
    # Patch in generator
    X.generator = CNSGAGenerator(vocs=X.vocs)
    X.max_evaluations = 5
    X.run()
    assert len(X.data) == 5
    assert all(~X.data["xopt_error"])
