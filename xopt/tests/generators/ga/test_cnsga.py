from random import random

import numpy as np
import os
import tempfile
import pandas as pd
import pytest

from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.ga.cnsga import CNSGAGenerator, uniform, cnsga_toolbox
from xopt.resources.test_functions.tnk import evaluate_TNK, tnk_vocs
from xopt.resources.test_functions.modified_tnk import (
    evaluate_modified_TNK,
    modified_tnk_vocs,
)
from xopt.stopping_conditions import MaxEvaluationsCondition


def test_cnsga():
    X = Xopt(
        generator=CNSGAGenerator(vocs=tnk_vocs),
        evaluator=Evaluator(function=evaluate_TNK),
        stopping_condition=MaxEvaluationsCondition(max_evaluations=10),
    )
    X.run()


def test_cnsga_single_objective():
    """
    Test for CNSGAGenerator single objective.
    """
    X = Xopt(
        generator=CNSGAGenerator(vocs=modified_tnk_vocs),
        evaluator=Evaluator(function=evaluate_modified_TNK),
        stopping_condition=MaxEvaluationsCondition(max_evaluations=5),
    )
    X.run()


def test_cnsga_baddf():
    # Test that bad dataframes are handled ok

    def eval_f(x):
        return {"y1": random(), "y2": random(), "c1": random(), "c2": random()}

    X = Xopt(
        generator=CNSGAGenerator(vocs=tnk_vocs, population_size=32),
        evaluator=Evaluator(function=eval_f),
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
    stopping_condition:
        name: MaxEvaluationsCondition
        max_evaluations: 10
    dump_file: null
    data: null
    generator:
        name: cnsga
        population_size: 8
        population_file: null
        vocs:
            variables:
                x1: [0, 3.14159]
                x2: [0, 3.14159]
            objectives: {y1: MINIMIZE, y2: MINIMIZE}
            constraints:
                c1: [GREATER_THAN, 0]
                c2: [LESS_THAN, 0.5]
            constants: {a: dummy_constant}

    evaluator:
        function: xopt.resources.test_functions.tnk.evaluate_TNK
        function_kwargs:
            sleep: 0
            random_sleep: 0.1


    """

    X = Xopt(YAML)
    # Patch in generator
    X.run()
    assert len(X.data) == 10
    assert all(~X.data["xopt_error"])


def test_cnsga_no_constraints():
    YAML = """
    stopping_condition:
        name: MaxEvaluationsCondition
        max_evaluations: 10
    dump_file: null
    data: null
    generator:
        name: cnsga
        population_size: 8
        population_file: null
        vocs:
            variables:
                x1: [0, 3.14159]
                x2: [0, 3.14159]
            objectives: {y1: MINIMIZE, y2: MINIMIZE}
            constraints: {}
            constants: {a: dummy_constant}

    evaluator:
        function: xopt.resources.test_functions.tnk.evaluate_TNK
        function_kwargs:
            sleep: 0
            random_sleep: 0.1


    """

    X = Xopt(YAML)
    # Patch in generator
    X.run()
    assert len(X.data) == 10
    assert all(~X.data["xopt_error"])


def test_write_offspring_and_population():
    gen = CNSGAGenerator(vocs=tnk_vocs, output_path=tempfile.gettempdir())
    # Create dummy offspring and population
    df = pd.DataFrame(
        {
            "x1": np.random.rand(5),
            "x2": np.random.rand(5),
            "y1": np.random.rand(5),
            "y2": np.random.rand(5),
            "c1": np.random.rand(5),
            "c2": np.random.rand(5),
        }
    )
    gen._offspring = df
    gen.population = df
    # Test write_offspring
    gen.write_offspring()
    # Test write_population
    gen.write_population()
    # Test with explicit filename
    off_file = os.path.join(tempfile.gettempdir(), "offspring_test.csv")
    pop_file = os.path.join(tempfile.gettempdir(), "population_test.csv")
    gen.write_offspring(off_file)
    gen.write_population(pop_file)
    assert os.path.exists(off_file)
    assert os.path.exists(pop_file)
    # Clean up
    os.remove(off_file)
    os.remove(pop_file)


def test_load_population_csv():
    gen = CNSGAGenerator(vocs=tnk_vocs)
    df = pd.DataFrame(
        {
            "x1": np.random.rand(5),
            "x2": np.random.rand(5),
            "y1": np.random.rand(5),
            "y2": np.random.rand(5),
            "c1": np.random.rand(5),
            "c2": np.random.rand(5),
        }
    )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index_label="xopt_index")
    tmp.close()
    gen.load_population_csv(tmp.name)
    assert hasattr(gen, "_loaded_population")
    assert gen._loaded_population is not None
    os.remove(tmp.name)


def test_uniform():
    # Test with lists
    low = [0, 0]
    up = [1, 1]
    result = uniform(low, up)
    assert len(result) == 2
    assert all(0 <= x <= 1 for x in result)
    # Test with scalars and size
    result2 = uniform(0, 1, size=3)
    assert len(result2) == 3
    assert all(0 <= x <= 1 for x in result2)


def test_cnsga_toolbox_options():
    from xopt.vocs import VOCS

    vocs = VOCS(
        variables={"x1": [0, 1], "x2": [0, 1]},
        objectives={"y1": "MINIMIZE", "y2": "MINIMIZE"},
        constraints={"c1": ["GREATER_THAN", 0]},
        constants={},
    )
    # Test default (auto/nsga2)
    tb = cnsga_toolbox(vocs)
    assert hasattr(tb, "select")
    # Test nsga2
    tb2 = cnsga_toolbox(vocs, selection="nsga2")
    assert hasattr(tb2, "select")
    # Test spea2
    tb3 = cnsga_toolbox(vocs, selection="spea2")
    assert hasattr(tb3, "select")
    # Test invalid
    with pytest.raises(ValueError):
        cnsga_toolbox(vocs, selection="invalid")
