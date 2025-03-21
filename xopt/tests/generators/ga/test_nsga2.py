import numpy as np
import pytest

from xopt.generators.ga.nsga2 import get_domination


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
