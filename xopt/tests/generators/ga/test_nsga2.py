import os
import json
from tempfile import TemporaryDirectory
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from xopt.base import Xopt
from xopt.errors import DataError
from xopt.evaluator import Evaluator
from xopt.generators.ga.nsga2 import (
    NSGA2Generator,
    generate_child_binary_tournament,
    crowded_comparison_argsort,
)
from xopt.generators.ga.operators import PolynomialMutation, SimulatedBinaryCrossover
from xopt.resources.test_functions.tnk import evaluate_TNK, tnk_vocs
from xopt.resources.test_functions.modified_tnk import (
    evaluate_modified_TNK,
    modified_tnk_vocs,
)

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


def test_nsga2_single_objective():
    """
    Test for NSGA2Generator single objective.
    """
    X = Xopt(
        generator=NSGA2Generator(vocs=modified_tnk_vocs),
        evaluator=Evaluator(function=evaluate_modified_TNK),
        vocs=modified_tnk_vocs,
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


@pytest.fixture
def nsga2_optimization_with_checkpoint():
    """Test fixture that supplies a optimization with data and the final checkpoint path"""
    with TemporaryDirectory() as output_dir:
        # Add a constant for testing VOCS reload with new parameters
        vocs = tnk_vocs.model_copy(deep=True)
        vocs.constants["my_const1"] = 0.0

        # Set up the generator with output_dir and checkpoint_freq
        generator = NSGA2Generator(
            vocs=vocs, output_dir=output_dir, population_size=10, checkpoint_freq=1
        )

        # Hack to avoid log error on windows: "The process cannot access the file because it is being used by another process"
        generator.ensure_output_dir_setup()
        generator.close_log_file()

        # Run a few optimization steps
        X = Xopt(
            generator=generator,
            evaluator=Evaluator(function=evaluate_TNK),
            vocs=vocs,
            max_evaluations=20,  # Run for 2 generations
        )
        X.run()

        # Verify that the checkpoint directory exists
        checkpoint_dir = os.path.join(output_dir, "checkpoints")

        # Get the latest checkpoint file
        checkpoint_files = os.listdir(checkpoint_dir)

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

        yield X, latest_checkpoint


def test_nsga2_checkpoint_reload_python(nsga2_optimization_with_checkpoint):
    """
    Test that NSGA2Generator can be reloaded from a checkpoint and used (python interface).
    """
    # Get the optimizer and checkpoint
    X, latest_checkpoint = nsga2_optimization_with_checkpoint

    # Create a new generator from the checkpoint
    restored_generator = NSGA2Generator(checkpoint_file=latest_checkpoint)

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


def test_nsga2_checkpoint_reload_yaml(nsga2_optimization_with_checkpoint):
    """
    Test that NSGA2Generator can be reloaded from a checkpoint and used (YAML interface).
    """
    # Get the optimizer and checkpoint
    X, latest_checkpoint = nsga2_optimization_with_checkpoint

    # Construct config file
    yaml = f"""
    max_evaluations: 20

    generator:
      name: nsga2
      checkpoint_file: {latest_checkpoint}

    evaluator:
      function: xopt.resources.test_functions.tnk.evaluate_TNK

    vocs:
      variables:
        x1: [0, 3.14159]
        x2: [0, 3.14159]

      objectives:
        y1: MINIMIZE
        y2: MINIMIZE

      constraints:
        c1: [GREATER_THAN, 0]
        c2: [LESS_THAN, 0.5]

      constants:
        a: dummy_constant
    """.replace("\n    ", "\n")

    # Reload from YAML, grab generator
    X_restored = Xopt.from_yaml(yaml)
    restored_generator = X_restored.generator

    # Verify that the restored generator has the same state as the original
    assert restored_generator.n_generations == X.generator.n_generations
    assert restored_generator.n_candidates == X.generator.n_candidates
    assert restored_generator.fevals == X.generator.fevals
    assert len(restored_generator.pop) == len(X.generator.pop)

    # Run a few more steps with the restored generator
    X_restored.run()

    # Verify that the restored generator continues from where it left off
    assert X_restored.generator.n_generations > X.generator.n_generations
    assert X_restored.generator.n_candidates > X.generator.n_candidates
    assert X_restored.generator.fevals > X.generator.fevals

    # Close log files before exiting context
    X.generator.close_log_file()
    if hasattr(X_restored.generator, "close_log_file"):
        X_restored.generator.close_log_file()


def test_nsga2_checkpoint_reload_override(nsga2_optimization_with_checkpoint):
    """
    Confirm that overriding settings works as intended
    """
    # Get the optimizer and checkpoint
    X, latest_checkpoint = nsga2_optimization_with_checkpoint

    # Create a new generator from the checkpoint
    new_pop_size = 20
    restored_generator = NSGA2Generator(
        checkpoint_file=latest_checkpoint, population_size=new_pop_size
    )

    # Check that the setting changed
    assert restored_generator.population_size == new_pop_size
    assert (
        X.generator.population_size != new_pop_size
    )  # Make sure we don't invalidate test in future by accident


def test_nsga2_checkpoint_reload_vocs_var_bounds_expand(
    nsga2_optimization_with_checkpoint,
):
    """
    Confirm we can expand bounds and individuals are not filtered.
    """
    my_xopt = Xopt.from_yaml(
        f"""
    generator:
      name: nsga2
      checkpoint_file: {nsga2_optimization_with_checkpoint[1]}

    evaluator:
      function: xopt.resources.test_functions.tnk.evaluate_TNK

    vocs:
      variables:
        x1: [-10.0, 10.0]
        x2: [-10.0, 10.0]

      objectives:
        y1: MINIMIZE
        y2: MINIMIZE

      constraints:
        c1: [GREATER_THAN, 0]
        c2: [LESS_THAN, 0.5]
    """.replace("\n    ", "\n")
    )

    # Check that all individuals are loaded (no filtering)
    orig_xopt = nsga2_optimization_with_checkpoint[0]
    assert len(my_xopt.generator.pop) == len(orig_xopt.generator.pop)
    assert len(my_xopt.generator.child) == len(orig_xopt.generator.child)


def test_nsga2_checkpoint_reload_vocs_var_bounds_shrink(
    nsga2_optimization_with_checkpoint,
):
    """
    Confirm we can change variable bounds and individuals outside are filtered.
    """
    my_xopt = Xopt.from_yaml(
        f"""
    generator:
      name: nsga2
      checkpoint_file: {nsga2_optimization_with_checkpoint[1]}

    evaluator:
      function: xopt.resources.test_functions.tnk.evaluate_TNK

    vocs:
      variables:
        x1: [-10.0, -5.0]
        x2: [-10.0, -5.0]

      objectives:
        y1: MINIMIZE
        y2: MINIMIZE

      constraints:
        c1: [GREATER_THAN, 0]
        c2: [LESS_THAN, 0.5]
    """.replace("\n    ", "\n")
    )

    # Check that all individuals are filtered (all out of bounds)
    assert len(my_xopt.generator.pop) == 0
    assert len(my_xopt.generator.child) == 0


def test_nsga2_checkpoint_reload_vocs_obj_dir(nsga2_optimization_with_checkpoint):
    Xopt.from_yaml(
        f"""
    generator:
      name: nsga2
      checkpoint_file: {nsga2_optimization_with_checkpoint[1]}

    evaluator:
      function: xopt.resources.test_functions.tnk.evaluate_TNK

    vocs:
      variables:
        x1: [0, 3.14159]
        x2: [0, 3.14159]

      objectives:
        y1: MAXIMIZE
        y2: MAXIMIZE

      constraints:
        c1: [GREATER_THAN, 0]
        c2: [LESS_THAN, 0.5]
    """.replace("\n    ", "\n")
    )


def test_nsga2_checkpoint_reload_vocs_constraint_conf(
    nsga2_optimization_with_checkpoint,
):
    Xopt.from_yaml(
        f"""
    generator:
      name: nsga2
      checkpoint_file: {nsga2_optimization_with_checkpoint[1]}

    evaluator:
      function: xopt.resources.test_functions.tnk.evaluate_TNK

    vocs:
      variables:
        x1: [0, 3.14159]
        x2: [0, 3.14159]

      objectives:
        y1: MINIMIZE
        y2: MINIMIZE

      constraints:
        c1: [LESS_THAN, 0.123]
        c2: [GREATER_THAN, 0.321]
    """.replace("\n    ", "\n")
    )


def test_nsga2_checkpoint_reload_vocs_new_var(nsga2_optimization_with_checkpoint):
    Xopt.from_yaml(
        f"""
    generator:
      name: nsga2
      checkpoint_file: {nsga2_optimization_with_checkpoint[1]}

    evaluator:
      function: xopt.resources.test_functions.tnk.evaluate_TNK

    vocs:
      variables:
        x1: [0, 3.14159]
        x2: [0, 3.14159]
        my_const1: [0.0, 1.0]

      objectives:
        y1: MINIMIZE
        y2: MINIMIZE

      constraints:
        c1: [GREATER_THAN, 0]
        c2: [LESS_THAN, 0.5]
    """.replace("\n    ", "\n")
    )


def test_nsga2_checkpoint_reload_vocs_new_obj(nsga2_optimization_with_checkpoint):
    Xopt.from_yaml(
        f"""
    generator:
      name: nsga2
      checkpoint_file: {nsga2_optimization_with_checkpoint[1]}

    evaluator:
      function: xopt.resources.test_functions.tnk.evaluate_TNK

    vocs:
      variables:
        x1: [0, 3.14159]
        x2: [0, 3.14159]

      objectives:
        y1: MINIMIZE
        y2: MINIMIZE
        my_const1: MINIMIZE

      constraints:
        c1: [GREATER_THAN, 0]
        c2: [LESS_THAN, 0.5]
    """.replace("\n    ", "\n")
    )


def test_nsga2_checkpoint_reload_vocs_new_const(nsga2_optimization_with_checkpoint):
    Xopt.from_yaml(
        f"""
    generator:
      name: nsga2
      checkpoint_file: {nsga2_optimization_with_checkpoint[1]}

    evaluator:
      function: xopt.resources.test_functions.tnk.evaluate_TNK

    vocs:
      variables:
        x1: [0, 3.14159]
        x2: [0, 3.14159]

      objectives:
        y1: MINIMIZE
        y2: MINIMIZE

      constraints:
        c1: [GREATER_THAN, 0]
        c2: [LESS_THAN, 0.5]
        my_const1: [LESS_THAN, 0.5]
    """.replace("\n    ", "\n")
    )


def test_nsga2_checkpoint_reload_vocs_bad_var(nsga2_optimization_with_checkpoint):
    with pytest.raises(ValueError, match="User-provided VOCS is not compatible.*"):
        Xopt.from_yaml(
            f"""
        generator:
          name: nsga2
          checkpoint_file: {nsga2_optimization_with_checkpoint[1]}

        evaluator:
          function: xopt.resources.test_functions.tnk.evaluate_TNK

        vocs:
          variables:
            x1: [0, 3.14159]
            x2: [0, 3.14159]
            does_not_exist: [0.0, 1.0]

          objectives:
            y1: MINIMIZE
            y2: MINIMIZE

          constraints:
            c1: [GREATER_THAN, 0]
            c2: [LESS_THAN, 0.5]
        """.replace("\n        ", "\n")
        )


def test_nsga2_checkpoint_reload_vocs_bad_obj(nsga2_optimization_with_checkpoint):
    with pytest.raises(ValueError, match="User-provided VOCS is not compatible.*"):
        Xopt.from_yaml(
            f"""
        generator:
          name: nsga2
          checkpoint_file: {nsga2_optimization_with_checkpoint[1]}

        evaluator:
          function: xopt.resources.test_functions.tnk.evaluate_TNK

        vocs:
          variables:
            x1: [0, 3.14159]
            x2: [0, 3.14159]

          objectives:
            y1: MINIMIZE
            y2: MINIMIZE
            does_not_exist: MINIMIZE

          constraints:
            c1: [GREATER_THAN, 0]
            c2: [LESS_THAN, 0.5]
        """.replace("\n        ", "\n")
        )


def test_nsga2_checkpoint_reload_vocs_bad_const(nsga2_optimization_with_checkpoint):
    with pytest.raises(ValueError, match="User-provided VOCS is not compatible.*"):
        Xopt.from_yaml(
            f"""
        generator:
          name: nsga2
          checkpoint_file: {nsga2_optimization_with_checkpoint[1]}

        evaluator:
          function: xopt.resources.test_functions.tnk.evaluate_TNK

        vocs:
          variables:
            x1: [0, 3.14159]
            x2: [0, 3.14159]

          objectives:
            y1: MINIMIZE
            y2: MINIMIZE

          constraints:
            c1: [GREATER_THAN, 0]
            c2: [LESS_THAN, 0.5]
            does_not_exist: [LESS_THAN, 0.5]
        """.replace("\n        ", "\n")
        )


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


def test_nsga2_output_inhomogenous_data():
    """
    Confirm that valid CSV files are written by generator when evaluator function doesn't have
    consistent schema.
    """
    with TemporaryDirectory() as output_dir:
        generator = NSGA2Generator(
            vocs=tnk_vocs,
            output_dir=output_dir,
            population_size=10,
        )

        # Hack to avoid log error on windows: "The process cannot access the file because it is being used by another process"
        generator.ensure_output_dir_setup()
        generator.close_log_file()

        # Run a few optimization steps
        X = Xopt(
            generator=generator,
            evaluator=Evaluator(function=evaluate_TNK),
            vocs=tnk_vocs,
            max_evaluations=30,  # Run for 3 generations
        )

        # Fill up a few populations with data
        for _ in range(3):
            for idx in range(X.generator.population_size):
                X.add_data(
                    pd.DataFrame(
                        {
                            "x1": np.random.random(),
                            "x2": np.random.random(),
                            "y1": np.random.random(),
                            "y2": np.random.random(),
                            "c1": np.random.random(),
                            "c2": np.random.random(),
                            "xopt_candidate_idx": idx,
                            "xopt_runtime": 0.1,
                            "xopt_error": False,
                        },
                        index=[0],
                    )
                )

        # Change the schema
        for _ in range(3):
            for _ in range(X.generator.population_size):
                X.add_data(
                    pd.DataFrame(
                        {
                            "x1": np.random.random(),
                            "x2": np.random.random(),
                            "y1": np.random.random(),
                            "y2": np.random.random(),
                            "c1": np.random.random(),
                            "c2": np.random.random(),
                            "obs1": np.random.random(),
                            "xopt_candidate_idx": idx,
                            "xopt_runtime": 0.1,
                            "xopt_error": False,
                        },
                        index=[0],
                    )
                )

        # Load the file (will fail if bad CSV file was written)
        pd.read_csv(os.path.join(output_dir, "populations.csv"))


def test_nsga2_vocs_not_present_in_add_data():
    # Run a few optimization steps
    X = Xopt(
        generator=NSGA2Generator(vocs=tnk_vocs),
        evaluator=Evaluator(function=evaluate_TNK),
        vocs=tnk_vocs,
        max_evaluations=10,
    )
    X.run()

    # Attempt to submit data with required vocs columns
    X.add_data(
        pd.DataFrame({"x1": [0], "x2": [0], "y1": [0], "y2": [0], "c1": [0], "c2": [0]})
    )

    # Missing var
    with pytest.raises(DataError, match="New data must contain at least all.*"):
        X.add_data(
            pd.DataFrame({"x1": [0], "y1": [0], "y2": [0], "c1": [0], "c2": [0]})
        )

    # Missing obj
    with pytest.raises(DataError, match="New data must contain at least all.*"):
        X.add_data(
            pd.DataFrame({"x1": [0], "x2": [0], "y1": [0], "c1": [0], "c2": [0]})
        )

    # Missing constraint
    with pytest.raises(DataError, match="New data must contain at least all.*"):
        X.add_data(
            pd.DataFrame({"x1": [0], "x2": [0], "y1": [0], "y2": [0], "c1": [0]})
        )

    # Try with strict=False
    X.strict = False
    X.add_data(pd.DataFrame({"x1": [0], "y2": [0], "c1": [0]}))
