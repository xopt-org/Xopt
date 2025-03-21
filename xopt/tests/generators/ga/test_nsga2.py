from random import random

import numpy as np
import pytest

from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.ga.nsga2 import get_domination, NSGA2Generator
from xopt.resources.test_functions.tnk import evaluate_TNK, tnk_vocs


@pytest.mark.parametrize(
    "pop_f, pop_g, expected_dom",
    [
        # Simple unconstrained case - individual 0 dominates 1
        (
            np.array([[1.0, 2.0], [2.0, 3.0]]),
            None,
            np.array([[False, True], [False, False]])
        ),
        
        # Non-dominated case
        (
            np.array([[1.0, 3.0], [2.0, 2.0]]),
            None,
            np.array([[False, False], [False, False]])
        ),
        
        # Constrained case - feasible dominates infeasible
        (
            np.array([[2.0, 2.0], [1.0, 1.0]]),
            np.array([[-1.0, -1.0], [1.0, -1.0]]),
            np.array([[False, True], [False, False]])
        ),
        
        # Both infeasible - less violation dominates
        (
            np.array([[2.0, 2.0], [1.0, 1.0]]),
            np.array([[1.0, 0.5], [2.0, 1.0]]),
            np.array([[False, True], [False, False]])
        ),
        
        # Three individuals with mixed domination
        (
            np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 3.0]]),
            None,
            np.array([
                [False, True, False],
                [False, False, False],
                [False, False, False]
            ])
        ),
        
        # Constrained case with three individuals
        (
            np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 0.5]]),
            np.array([[-1.0, -1.0], [-1.0, -1.0], [1.0, 1.0]]),
            np.array([
                [False, True, True],
                [False, False, True],
                [False, False, False]
            ])
        ),
    ]
)
def test_get_domination(pop_f, pop_g, expected_dom):
    """
    Test the get_domination function with various scenarios.
    """
    result = get_domination(pop_f, pop_g)
    np.testing.assert_array_equal(result, expected_dom)


def test_nsga2():
    """
    Basic test for NSGA2Generator.
    """
    X = Xopt(
        generator=NSGA2Generator(vocs=tnk_vocs),
        evaluator=Evaluator(function=evaluate_TNK),
        vocs=tnk_vocs,
        max_evaluations=5,
    )
    X.run()
