import os
import json
from tempfile import TemporaryDirectory
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.ga.nsga2 import (
    NSGA2Generator,
    generate_child_binary_tournament,
    crowded_comparison_argsort,
)
from xopt.generators.ga.operators import PolynomialMutation, SimulatedBinaryCrossover
from xopt.resources.test_functions.tnk import evaluate_TNK, tnk_vocs
from xopt.vocs import VOCS


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


def test_nsga2_output_data():
    """
    Test that NSGA2Generator correctly outputs data to files.
    Uses a temporary directory to avoid polluting the filesystem.
    """
    with TemporaryDirectory() as output_dir:
        generator = NSGA2Generator(
            vocs=tnk_vocs,
            output_dir=output_dir,
            population_size=10,
        )

        # Run a few optimization steps
        X = Xopt(
            generator=generator,
            evaluator=Evaluator(function=evaluate_TNK),
            vocs=tnk_vocs,
            max_evaluations=30,  # Run for 3 generations
        )
        X.run()

        # Verify that the data files are created
        assert os.path.exists(os.path.join(output_dir, "data.csv"))
        assert os.path.exists(os.path.join(output_dir, "populations.csv"))
        assert os.path.exists(os.path.join(output_dir, "vocs.txt"))
        assert os.path.exists(os.path.join(output_dir, "log.txt"))

        # Read the data file and check its contents
        data_df = pd.read_csv(os.path.join(output_dir, "data.csv"))

        # Check that the data file contains the expected number of rows
        assert len(data_df) == 30

        # Check that the data file contains the expected columns
        expected_columns = [
            "x1",
            "x2",
            "y1",
            "y2",
            "c1",
            "c2",
            "xopt_runtime",
            "xopt_error",
            "xopt_parent_generation",
            "xopt_candidate_idx",
        ]
        for col in expected_columns:
            assert col in data_df.columns

        # Read the populations file and check its contents
        pop_df = pd.read_csv(os.path.join(output_dir, "populations.csv"))

        # Check that the populations file contains the expected number of rows
        assert len(pop_df) == 30

        # Check that the populations file contains the expected columns
        assert "xopt_generation" in pop_df.columns

        # Check that the VOCS file contains valid JSON
        with open(os.path.join(output_dir, "vocs.txt"), "r") as f:
            vocs_dict = json.load(f)
            VOCS.from_dict(vocs_dict)

        # Verify that the log file exists and has content
        with open(os.path.join(output_dir, "log.txt"), "r") as f:
            log_content = f.read()
            assert len(log_content) > 0

        # Close log file before exiting context
        X.generator.close_log_file()


def test_generate_child_binary_tournament():
    """
    Test that generate_child_binary_tournament function works correctly with random input.

    This test verifies that:
    1. The function runs without errors with random input
    2. The output is within the specified bounds
    3. The output doesn't contain NaN values
    """
    # Create random population data
    n_individuals = 10
    n_variables = 3
    n_objectives = 2
    n_constraints = 2

    # Create random population with decision variables
    pop_x = np.random.rand(n_individuals, n_variables)

    # Create random objective values (lower is better)
    pop_f = np.random.rand(n_individuals, n_objectives)

    # Create random constraint values (<=0 means satisfied)
    pop_g = np.random.uniform(-1, 1, (n_individuals, n_constraints))

    # Define bounds for variables
    lower_bounds = np.zeros(n_variables)
    upper_bounds = np.ones(n_variables)
    bounds = np.vstack((lower_bounds, upper_bounds))

    # Create mutation and crossover operators
    mutation = PolynomialMutation()
    crossover = SimulatedBinaryCrossover()

    # Generate a child using binary tournament
    child = generate_child_binary_tournament(
        pop_x=pop_x,
        pop_f=pop_f,
        pop_g=pop_g,
        bounds=bounds,
        mutate=mutation,
        crossover=crossover,
    )

    # Verify the child has the correct shape
    assert child.shape == (n_variables,), (
        f"Expected shape {(n_variables,)}, got {child.shape}"
    )

    # Verify the child is within bounds
    assert np.all(child >= lower_bounds), "Child contains values below lower bounds"
    assert np.all(child <= upper_bounds), "Child contains values above upper bounds"

    # Verify the child doesn't contain NaN values
    assert not np.any(np.isnan(child)), "Child contains NaN values"

    # Test with pre-computed fitness
    fitness = np.random.randint(0, n_individuals, n_individuals)
    child_with_fitness = generate_child_binary_tournament(
        pop_x=pop_x,
        pop_f=pop_f,
        pop_g=pop_g,
        bounds=bounds,
        mutate=mutation,
        crossover=crossover,
        fitness=fitness,
    )

    # Verify the child with fitness has the correct shape
    assert child_with_fitness.shape == (n_variables,), (
        f"Expected shape {(n_variables,)}, got {child_with_fitness.shape}"
    )

    # Verify the child with fitness is within bounds
    assert np.all(child_with_fitness >= lower_bounds), (
        "Child with fitness contains values below lower bounds"
    )
    assert np.all(child_with_fitness <= upper_bounds), (
        "Child with fitness contains values above upper bounds"
    )

    # Verify the child with fitness doesn't contain NaN values
    assert not np.any(np.isnan(child_with_fitness)), (
        "Child with fitness contains NaN values"
    )


def test_nsga2_checkpoint_reload():
    """
    Test that NSGA2Generator can be reloaded from a checkpoint.
    """
    with TemporaryDirectory() as output_dir:
        # Set up the generator with output_dir and checkpoint_freq
        generator = NSGA2Generator(
            vocs=tnk_vocs, output_dir=output_dir, population_size=10, checkpoint_freq=1
        )

        # Run a few optimization steps
        X = Xopt(
            generator=generator,
            evaluator=Evaluator(function=evaluate_TNK),
            vocs=tnk_vocs,
            max_evaluations=20,  # Run for 2 generations
        )
        X.run()

        # Verify that the checkpoint directory exists
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        assert os.path.exists(checkpoint_dir)

        # Get the latest checkpoint file
        checkpoint_files = os.listdir(checkpoint_dir)
        assert len(checkpoint_files) >= 1

        def get_checkpoint_datetime(filename):
            # Extract the date part and deduplication index
            parts = filename.rsplit("_", 1)
            date_part = parts[0]
            index = int(parts[1].split(".")[0] if len(parts) > 1 else "0")

            # Return tuple of (datetime, index) for lexical sorting
            ret = (datetime.strptime(date_part, "%Y%m%d_%H%M%S"), index)
            return ret

        sorted_checkpoints = sorted(checkpoint_files, key=get_checkpoint_datetime)
        latest_checkpoint = os.path.join(checkpoint_dir, sorted_checkpoints[-1])

        # Load the generator from the checkpoint
        with open(latest_checkpoint, "r") as f:
            checkpoint_data = json.load(f)

        # Create a new generator from the checkpoint
        restored_generator = NSGA2Generator.from_dict(
            {"vocs": tnk_vocs, **checkpoint_data}
        )

        # Verify that the restored generator has the same state as the original
        assert restored_generator.n_generations == X.generator.n_generations
        assert restored_generator.n_candidates == X.generator.n_candidates
        assert restored_generator.fevals == X.generator.fevals
        assert len(restored_generator.pop) == len(X.generator.pop)

        # Run a few more steps with the restored generator
        X_restored = Xopt(
            generator=restored_generator,
            evaluator=Evaluator(function=evaluate_TNK),
            vocs=tnk_vocs,
            max_evaluations=10,  # Run for 1 more generation
        )
        X_restored.run()

        # Verify that the restored generator continues from where it left off
        assert X_restored.generator.n_generations > X.generator.n_generations
        assert X_restored.generator.n_candidates > X.generator.n_candidates
        assert X_restored.generator.fevals > X.generator.fevals

        # Close log files before exiting context
        X.generator.close_log_file()
        if hasattr(X_restored.generator, "close_log_file"):
            X_restored.generator.close_log_file()


def test_nsga2_all_individuals_in_data():
    """
    Test that all individuals generated by NSGA2Generator are included in the data file.
    """
    with TemporaryDirectory() as output_dir:
        generator = NSGA2Generator(
            deduplicate_output=False,
            vocs=tnk_vocs,
            output_dir=output_dir,
            population_size=10,
        )

        # Run a few optimization steps
        X = Xopt(
            generator=generator,
            evaluator=Evaluator(function=evaluate_TNK, max_workers=1),
            vocs=tnk_vocs,
        )
        for _ in range(30):
            X.step()

        # Read the data file
        data_df = pd.read_csv(os.path.join(output_dir, "data.csv"))

        # Get all candidate indices from the generator's history
        all_history_indices = []
        for gen_indices in X.generator.history_idx:
            all_history_indices.extend(gen_indices)

        # Check that all individuals in the history are in the data file
        for idx in all_history_indices:
            assert idx in data_df["xopt_candidate_idx"].values

        # Check that the number of unique candidate indices in the data file
        # matches the generator's n_candidates counter
        assert len(data_df["xopt_candidate_idx"]) == X.generator.n_candidates

        # Close log file before exiting context
        X.generator.close_log_file()


def test_resume_consistency(pop_size=5, n_steps=128, check_step=10):
    """
    Test that NSGA2Generator produces consistent results when serialized and deserialized.

    This test verifies that:
    1. The generator produces the same results when reloaded from a serialized state
    2. The random seed ensures deterministic behavior

    Because NSGA2 is stochastic, we set a fixed random seed before each operation
    to ensure reproducibility.
    """
    # Use TNK problem for testing
    problem_vocs = tnk_vocs
    problem_func = evaluate_TNK

    # Function to compare two Xopt objects
    def compare(val_a, val_b):
        """Compare two Xopt objects"""
        # Must ignore runtime
        for x in val_a.generator.child:
            x["xopt_runtime"] = 0.0
        for x in val_b.generator.child:
            x["xopt_runtime"] = 0.0
        for x in val_a.generator.pop:
            x["xopt_runtime"] = 0.0
        for x in val_b.generator.pop:
            x["xopt_runtime"] = 0.0

        # Compare everything except data
        y = json.loads(val_a.json())
        y2 = json.loads(val_b.json())
        y.pop("data")
        y2.pop("data")
        assert y == y2

        # Compare the data (taken from neldermead tests)
        # For unclear reasons, column order changes on reload....
        data = X.data.drop(["xopt_runtime", "xopt_error"], axis=1)
        data2 = X2.data.drop(["xopt_runtime", "xopt_error"], axis=1)
        # On reload, index is not a range index anymore!
        pd.testing.assert_frame_equal(data, data2, check_index_type=False)

    # Create the Xopt object
    X = Xopt(
        generator=NSGA2Generator(
            vocs=problem_vocs,
            population_size=pop_size,
        ),
        evaluator=Evaluator(function=problem_func),
        vocs=problem_vocs,
    )

    # Run the first step to initialize
    np.random.seed(42)
    X.step()

    # Run through steps, checking consistency at intervals
    for i in range(1, n_steps):
        # For performance, only check some steps
        if i % check_step == 0 or i == n_steps - 1:
            # Serialize the current state
            state = X.json()
            X2 = Xopt.model_validate(json.loads(state))
            compare(X, X2)

            # Generate samples from both and compare
            np.random.seed(42)
            samples = X.generator.generate(1)
            np.random.seed(42)
            samples2 = X2.generator.generate(1)

            assert samples == samples2, f"Generated samples differ at step {i}"

            # Evaluate the samples
            np.random.seed(42)
            X.evaluate_data(samples)
            X2.evaluate_data(samples2)
            compare(X, X2)

            # Create a third instance to test another serialization cycle and compare to X2
            X3 = Xopt.model_validate(json.loads(state))
            np.random.seed(42)
            samples3 = X3.generator.generate(1)
            assert samples == samples3, (
                f"Generated samples differ in third instance at step {i}"
            )
            X3.evaluate_data(samples3)
            compare(X2, X3)
        else:
            # Just run a normal step
            np.random.seed(42)
            samples = X.generator.generate(1)
            X.evaluate_data(samples)


@pytest.mark.parametrize(
    "pop_f, pop_g, expected_indices_options",
    [
        # Two individuals in different ranks
        (np.array([[1.0, 2.0], [2.0, 3.0]]), None, [np.array([1, 0])]),
        # Non-dominated, different crowding distances
        (
            np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]),
            None,
            [np.array([1, 2, 0]), np.array([1, 0, 2])],
        ),
        # NaN values
        (
            np.array([[1.0, 2.0], [np.nan, 3.0], [2.0, 1.0]]),
            None,
            [np.array([1, 2, 0]), np.array([1, 0, 2])],
        ),
        # Constrained
        (
            np.array([[2.0, 2.0], [1.0, 1.0]]),
            np.array([[-1.0, -1.0], [1.0, -1.0]]),
            [np.array([1, 0])],
        ),
        # Multiple individuals with same rank but potentially same crowding distances
        (
            np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [1.5, 2.5]]),
            None,
            [
                np.array([3, 1, 2, 0]),
                np.array([1, 3, 2, 0]),
                np.array([3, 1, 0, 2]),
                np.array([1, 3, 0, 2]),
            ],
        ),
        # NaN values and constraints
        (
            np.array([[1.0, 2.0], [np.nan, 3.0], [2.0, 1.0]]),
            np.array([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            [np.array([1, 2, 0])],
        ),
        # All NaN
        (
            np.array([[np.nan, np.nan], [np.nan, np.nan]]),
            None,
            [np.array([0, 1]), np.array([1, 0])],
        ),
    ],
)
def test_crowded_comparison_argsort(pop_f, pop_g, expected_indices_options):
    """
    Test the crowded_comparison_argsort function with various explicit input.

    Parameters
    ----------
    pop_f : numpy.ndarray
        Objective values
    pop_g : numpy.ndarray or None
        Constraint values
    expected_indices_options : list of numpy.ndarray
        List of valid expected sorted indices
    """
    # Call the function
    result = crowded_comparison_argsort(pop_f, pop_g)

    # Check if the result matches any of the expected options
    matches_any = any(
        np.array_equal(result, expected) for expected in expected_indices_options
    )

    if not matches_any:
        message = (
            f"Result {result} doesn't match any expected ordering.\n"
            f"Expected one of: {expected_indices_options}"
        )
        assert False, message
