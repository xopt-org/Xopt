import numpy as np
import pytest

from xopt.generators.ga.operators import (
    PolynomialMutation,
    SimulatedBinaryCrossover,
)


@pytest.mark.parametrize(
    "mutation_operator",
    [
        PolynomialMutation(),
        PolynomialMutation(pm=0.1),
        PolynomialMutation(eta_m=10),
    ],
)
def test_mutation_operator(mutation_operator):
    """
    Test mutation operators by running them multiple times with random inputs.

    Verifies that:
    1. Output has the correct shape
    2. Output does not contain NaN values
    3. Output is within bounds
    """
    # Number of test runs
    n_runs = 32

    # Test with different dimensions
    for n_dims in [2, 5, 10]:
        for _ in range(n_runs):
            # Create random bounds
            lower_bounds = np.random.uniform(-10, 0, n_dims)
            upper_bounds = np.random.uniform(1, 10, n_dims)
            bounds = np.vstack([lower_bounds, upper_bounds])

            # Create random parent within bounds
            parent = np.random.uniform(lower_bounds, upper_bounds, n_dims)

            # Apply mutation
            child = mutation_operator(parent, bounds)

            # Check shape
            assert child.shape == parent.shape, (
                f"Output shape {child.shape} doesn't match input shape {parent.shape}"
            )

            # Check for NaN values
            assert not np.isnan(child).any(), "Output contains NaN values"

            # Check bounds
            assert np.all(child >= bounds[0]), (
                "Output contains values below lower bounds"
            )
            assert np.all(child <= bounds[1]), (
                "Output contains values above upper bounds"
            )


@pytest.mark.parametrize(
    "crossover_operator",
    [
        SimulatedBinaryCrossover(),
        SimulatedBinaryCrossover(delta_1=0.8),
        SimulatedBinaryCrossover(delta_2=0.8),
        SimulatedBinaryCrossover(eta_c=10),
    ],
)
def test_crossover_operator(crossover_operator):
    """
    Test crossover operators by running them multiple times with random inputs.

    Verifies that:
    1. Output has the correct shape
    2. Output does not contain NaN values
    3. Output is within bounds
    """
    # Number of test runs
    n_runs = 32

    # Test with different dimensions
    for n_dims in [2, 5, 10]:
        for _ in range(n_runs):
            # Create random bounds
            lower_bounds = np.random.uniform(-10, 0, n_dims)
            upper_bounds = np.random.uniform(1, 10, n_dims)
            bounds = np.vstack([lower_bounds, upper_bounds])

            # Create random parents within bounds
            parent_a = np.random.uniform(lower_bounds, upper_bounds, n_dims)
            parent_b = np.random.uniform(lower_bounds, upper_bounds, n_dims)

            # Apply crossover
            child_a, child_b = crossover_operator(parent_a, parent_b, bounds)

            # Check shape
            assert child_a.shape == parent_a.shape, (
                f"Child A shape {child_a.shape} doesn't match parent shape {parent_a.shape}"
            )
            assert child_b.shape == parent_b.shape, (
                f"Child B shape {child_b.shape} doesn't match parent shape {parent_b.shape}"
            )

            # Check for NaN values
            assert not np.isnan(child_a).any(), "Child A contains NaN values"
            assert not np.isnan(child_b).any(), "Child B contains NaN values"

            # Check bounds
            assert np.all(child_a >= bounds[0]), (
                "Child A contains values below lower bounds"
            )
            assert np.all(child_a <= bounds[1]), (
                "Child A contains values above upper bounds"
            )
            assert np.all(child_b >= bounds[0]), (
                "Child B contains values below lower bounds"
            )
            assert np.all(child_b <= bounds[1]), (
                "Child B contains values above upper bounds"
            )
