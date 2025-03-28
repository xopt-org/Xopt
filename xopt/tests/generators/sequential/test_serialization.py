import pickle

import numpy as np
import pytest
from xopt.generators.sequential import (
    RCDSGenerator,
    ExtremumSeekingGenerator,
    NelderMeadGenerator,
)
from xopt import Evaluator, Xopt
from xopt.vocs import VOCS


def sin_function(input_dict):
    x = input_dict["x"]
    return {
        "f": -10 * np.exp(-((x - np.pi) ** 2) / 0.01) + 0.5 * np.sin(5 * x),
    }


class TestSequentialSerialization:
    @pytest.mark.parametrize(
        "generator", [RCDSGenerator, ExtremumSeekingGenerator, NelderMeadGenerator]
    )
    def test_serialization_and_restart(self, generator):
        test_vocs = VOCS(
            variables={"x": [0, 3.14159]},
            objectives={"f": "MINIMIZE"},
            constants={},
        )
        evaluator = Evaluator(function=sin_function)
        gen = generator(vocs=test_vocs)

        X = Xopt(vocs=test_vocs, evaluator=evaluator, generator=gen)

        X.random_evaluate(1)
        for i in range(10):
            X.step()

        dump = X.yaml()
        X2 = Xopt.from_yaml(dump)

        for i in range(10):
            X2.step()

        # test pickling
        pickle.dumps(X2.generator)
