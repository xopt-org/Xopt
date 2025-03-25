import numpy as np
import pytest

from xopt.generators.utils import (
    get_domination,
    fast_dominated_argsort_internal,
)


@pytest.mark.parametrize(
    "pop_f, pop_g, expected_dom",
    [
        # Simple unconstrained case - individual 0 dominates 1
        (
            np.array([[1.0, 2.0], [2.0, 3.0]]),
            None,
            np.array([[False, True], [False, False]]),
        ),
        # Non-dominated case
        (
            np.array([[1.0, 3.0], [2.0, 2.0]]),
            None,
            np.array([[False, False], [False, False]]),
        ),
        # Constrained case - feasible dominates infeasible
        (
            np.array([[2.0, 2.0], [1.0, 1.0]]),
            np.array([[-1.0, -1.0], [1.0, -1.0]]),
            np.array([[False, True], [False, False]]),
        ),
        # Both infeasible - less violation dominates
        (
            np.array([[2.0, 2.0], [1.0, 1.0]]),
            np.array([[1.0, 0.5], [2.0, 1.0]]),
            np.array([[False, True], [False, False]]),
        ),
        # Three individuals with mixed domination
        (
            np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 3.0]]),
            None,
            np.array(
                [[False, True, False], [False, False, False], [False, False, False]]
            ),
        ),
        # Constrained case with three individuals
        (
            np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 0.5]]),
            np.array([[-1.0, -1.0], [-1.0, -1.0], [1.0, 1.0]]),
            np.array(
                [[False, True, True], [False, False, True], [False, False, False]]
            ),
        ),
    ],
)
def test_get_domination(pop_f, pop_g, expected_dom):
    """
    Test the get_domination function with various scenarios.
    """
    result = get_domination(pop_f, pop_g)
    np.testing.assert_array_equal(result, expected_dom)


@pytest.mark.parametrize(
    "dom, expected_ranks",
    [
        # Test case 1: Simple domination chain
        (
            np.array(
                [
                    [False, True, False, False],
                    [False, False, True, False],
                    [False, False, False, True],
                    [False, False, False, False],
                ]
            ),
            [
                [0],
                [1],
                [2],
                [3],
            ],
        ),
        # Test case 2: Multiple individuals in the same front
        (
            np.array(
                [
                    [False, False, True, True],
                    [False, False, True, True],
                    [False, False, False, False],
                    [False, False, False, False],
                ]
            ),
            [
                [0, 1],
                [2, 3],
            ],
        ),
        # Test case 3: Multiple fronts
        (
            np.array(
                [
                    [False, False, True, True, True, False],
                    [False, False, False, True, True, False],
                    [False, False, False, False, True, True],
                    [False, False, False, False, False, True],
                    [False, False, False, False, False, False],
                    [False, False, False, False, False, False],
                ]
            ),
            [
                [0, 1],
                [2, 3],
                [4, 5],
            ],
        ),
    ],
)
def test_fast_dominated_argsort_internal(dom, expected_ranks):
    """
    Test the fast_dominated_argsort_internal function with various domination matrices.

    This function tests the core nondominated sorting algorithm used in NSGA-II
    with specific domination matrices and verifies the correct sorting of individuals
    into domination ranks.

    Parameters
    ----------
    dom : numpy.ndarray
        Boolean domination matrix where dom[i,j] = True means individual i dominates j
    expected_ranks : list of lists
        Expected sorting of individuals into domination ranks
    """
    result = fast_dominated_argsort_internal(dom)

    # Check that we have the expected number of ranks
    assert len(result) == len(expected_ranks), (
        f"Expected {len(expected_ranks)} ranks, got {len(result)}"
    )

    # Check each rank contains the expected individuals
    for i, (result_rank, expected_rank) in enumerate(zip(result, expected_ranks)):
        # Convert result to list for easier comparison
        result_rank_list = (
            result_rank.tolist() if isinstance(result_rank, np.ndarray) else result_rank
        )

        # Sort both lists to handle different ordering within the same rank
        assert sorted(result_rank_list) == sorted(expected_rank), (
            f"Rank {i} mismatch: expected {expected_rank}, got {result_rank_list}"
        )
