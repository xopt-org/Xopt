import pytest
from xopt.generators.deduplicated import DeduplicatedGeneratorBase
from xopt.resources.test_functions.zdt import construct_zdt


class DummyGenerator(DeduplicatedGeneratorBase):
    """
    A test generator that intentionally produces duplicate values.

    This generator creates candidates where every other output is a duplicate
    of a previous output, allowing us to test the deduplication functionality.
    """

    name = "dummy_generator"
    supports_multi_objective: bool = True
    _counter: int = 0

    def _generate(self, n_candidates: int) -> list[dict]:
        """
        Generate candidates with intentional duplicates.

        Every even-indexed candidate will be a duplicate of the previous
        odd-indexed candidate.

        Parameters
        ----------
        n_candidates : int
            Number of candidates to generate.

        Returns
        -------
        list of dict
            List of candidate solutions, with duplicates.
        """
        candidates = []

        for i in range(n_candidates):
            # For even indices (except 0), duplicate the previous candidate
            if i > 0 and i % 2 == 0 and candidates:
                candidates.append(candidates[-1].copy())
            else:
                # Create a new unique candidate with incrementing values
                candidate = {}
                for j in range(self.vocs.n_variables):
                    var_name = f"x{j + 1}"
                    # Use a smaller multiplier to ensure uniqueness for tens of candidates
                    value = (self._counter * 0.001 + j * 0.0001) % 1.0
                    candidate[var_name] = value

                self._counter += 1
                candidates.append(candidate)

        return candidates


@pytest.fixture
def zdt_vocs():
    """Fixture to provide a ZDT1 VOCS with 5 dimensions."""
    vocs, _, _ = construct_zdt(n_dims=5, problem_index=1)
    return vocs


def test_with_deduplication(zdt_vocs):
    """Test that duplicates are removed when deduplication is enabled."""
    # Create generator with deduplication enabled
    generator = DummyGenerator(vocs=zdt_vocs, deduplicate_output=True)

    # Generate 10 candidates - should result in ~5 unique ones due to duplication
    candidates = generator.generate(10)

    # Check that we got 10 candidates (the generator should have made extra calls to get enough unique values)
    assert len(candidates) == 10

    # Convert candidates to tuples of their values for comparison
    candidate_tuples = [tuple(c.values()) for c in candidates]

    # Check that all candidates are unique
    assert len(candidate_tuples) == len(set(candidate_tuples))

    # Check that the values follow the expected pattern (increasing counter values)
    for i in range(len(candidates) - 1):
        # Compare the first variable (x1) which should be increasing
        assert candidates[i]["x1"] <= candidates[i + 1]["x1"]


def test_without_deduplication(zdt_vocs):
    """Test that duplicates are preserved when deduplication is disabled."""
    # Create generator with deduplication disabled
    generator = DummyGenerator(vocs=zdt_vocs, deduplicate_output=False)

    # Generate 10 candidates
    candidates = generator.generate(10)

    # Check that we got 10 candidates
    assert len(candidates) == 10

    # Check the actual values match the formula used to generate them
    # The formula is: value = (counter * 0.001 + j * 0.0001) % 1.0

    # For non-duplicate candidates (indices 0, 1, 3, 5, 7, 9), check the values
    expected_counter = 0
    for i in range(len(candidates)):
        # Skip even indices (except 0) as they are duplicates
        if i > 0 and i % 2 == 0:
            # Check that this candidate is a duplicate of the previous one
            assert candidates[i] == candidates[i - 1], (
                f"Expected candidate at index {i} to be a duplicate of candidate at index {i - 1}"
            )
        else:
            # Check that the values match the formula
            for j in range(zdt_vocs.n_variables):
                var_name = f"x{j + 1}"
                expected_value = (expected_counter * 0.001 + j * 0.0001) % 1.0
                assert abs(candidates[i][var_name] - expected_value) < 1e-10, (
                    f"Value mismatch at index {i}, variable {var_name}: expected {expected_value}, got {candidates[i][var_name]}"
                )
            expected_counter += 1


def test_decision_vars_seen_persistence(zdt_vocs):
    """Test that decision_vars_seen persists between generate calls."""
    # Create generator with deduplication enabled
    generator = DummyGenerator(vocs=zdt_vocs, deduplicate_output=True)

    # First generation
    candidates1 = generator.generate(5)

    # Second generation
    candidates2 = generator.generate(5)

    # Convert all candidates to tuples for comparison
    all_candidate_tuples = [tuple(c.values()) for c in candidates1 + candidates2]

    # Check that all values across both generations are unique
    assert len(all_candidate_tuples) == len(set(all_candidate_tuples))

    # Check that the second batch continues from where the first left off
    # by comparing the first variable (x1) of the last candidate in first batch
    # and the first candidate in second batch
    assert candidates1[-1]["x1"] < candidates2[0]["x1"]
