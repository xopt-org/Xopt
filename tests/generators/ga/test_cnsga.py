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
        max_evaluations=5,
    )
    X.run()


def test_cnsga_from_yaml():
    YAML = """
    max_evaluations: 5
    dump_file: null
    data: null
    generator:
        name: cnsga
        population_size: 64
        population_file: null  # Bad
      
    evaluator:
        function: xopt.resources.test_functions.tnk.evaluate_TNK
        function_kwargs:
            sleep: 0
            random_sleep: 0.1
      
    vocs:
        variables:
            x1: [0, 3.14159]
            x2: [0, 3.14159]
        objectives: {y1: MINIMIZE, y2: MINIMIZE}
        constraints:
            c1: [GREATER_THAN, 0]
            c2: [LESS_THAN, 0.5]
        constants: {a: dummy_constant}
    """

    X = Xopt(YAML)
    # Patch in generator
    X.run()
    assert len(X.data) == 5
    assert all(~X.data["xopt_error"])
