from random import random

import numpy as np

from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.ga.cnsga import CNSGAGenerator
from xopt.resources.test_functions.tnk import evaluate_TNK, tnk_vocs


def test_cnsga():
    X = Xopt(
        generator=CNSGAGenerator(vocs=tnk_vocs),
        evaluator=Evaluator(function=evaluate_TNK),
        vocs=tnk_vocs,
        max_evaluations=5,
    )
    X.run()


def test_cnsga_baddf():
    # Test that bad dataframes are handled ok

    def eval_f(x):
        return {"y1": random(), "y2": random(), "c1": random(), "c2": random()}

    X = Xopt(
        generator=CNSGAGenerator(vocs=tnk_vocs, population_size=32),
        evaluator=Evaluator(function=eval_f),
        vocs=tnk_vocs,
        strict=False,
    )

    X.random_evaluate(12)

    for i in range(100):
        new_samples = X.generator.generate(1)
        X.evaluate_data(new_samples)

    assert len(X.generator.population) == 32

    for i in range(100):
        new_samples = X.generator.generate(12)
        X.evaluate_data(new_samples[0:2])
        X.evaluate_data(new_samples[2:4])
        X.evaluate_data(new_samples[4:12])

    assert len(X.generator.population) == 32

    for i in range(10):
        X.generator.generate(20)
        bad_df = X.data.iloc[np.random.randint(0, 200, 20), :].copy()
        bad_df.index = np.ones(20, dtype=int)
        X.generator.add_data(bad_df)

    for i in range(20):
        X.step()

    assert len(X.generator.population) == 32


def test_cnsga_from_yaml():
    YAML = """
    max_evaluations: 10
    dump_file: null
    data: null
    generator:
        name: cnsga
        population_size: 8
        population_file: null

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
    assert len(X.data) == 10
    assert all(~X.data["xopt_error"])


def test_cnsga_no_constraints():
    YAML = """
    max_evaluations: 10
    dump_file: null
    data: null
    generator:
        name: cnsga
        population_size: 8
        population_file: null

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
        constraints: {}
        constants: {a: dummy_constant}
    """

    X = Xopt(YAML)
    # Patch in generator
    X.run()
    assert len(X.data) == 10
    assert all(~X.data["xopt_error"])
