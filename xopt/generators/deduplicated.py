import numpy as np
from xopt.generator import Generator
from pydantic import field_validator
from typing import Optional
import logging
import time


class DeduplicatedGeneratorBase(Generator):
    """
    Base class for generators that avoid producing duplicate candidates.

    Parameters
    ----------
    deduplicate_output : bool, default=True
        Whether to perform deduplication on generated candidates.
    decision_vars_seen : numpy.ndarray, optional
        Array of previously seen decision variables, shape (n_seen, n_variables).
        If None, will be initialized on first generation.

    Notes
    -----
    Subclasses must implement the `_generate` method which produces
    candidate solutions. The base class handles the deduplication logic.

    Deduplication is performed using numpy's `unique` function to identify
    and filter out duplicate decision vectors. The class maintains a history
    of all previously seen decision variables to ensure global uniqueness
    across multiple generate calls.
    """

    # Whether to perform deduplication or not
    deduplicate_output: bool = True

    # The decision vars seen so far
    decision_vars_seen: Optional[np.ndarray] = None

    # For per-object log output in child objects (see eg NSGA2Generator)
    _logger: Optional[logging.Logger] = None

    def model_post_init(self, context):
        # Get a unique logger per object
        self._logger = logging.getLogger(
            f"{__name__}.DeduplicatedGeneratorBase.{id(self)}"
        )

    @field_validator("decision_vars_seen", mode="before")
    @classmethod
    def cast_arr(cls, value):
        if isinstance(value, list):
            return np.array(value)
        return value

    def generate(self, n_candidates: int) -> list[dict]:
        """
        Generate the unique candidates.

        If deduplication is enabled, ensures all returned candidates have
        unique decision variables that have not been seen before.

        Parameters
        ----------
        n_candidates : int
            Number of unique candidates to generate.

        Returns
        -------
        list of dict
            List of candidate solutions.

        Notes
        -----
        When deduplication is enabled, the method may make multiple calls
        to the underlying `_generate` method if duplicates are found, until
        the requested number of unique candidates is obtained.
        """
        start_t = time.perf_counter()
        if not self.deduplicate_output:
            candidates = self._generate(n_candidates)
            n_removed = 0
        else:
            # Create never before seen candidates by calling child generator and only taking unique
            # value from it until we have `n_candidates` values.
            candidates = []
            n_removed = 0
            round_idx = 0
            while len(candidates) < n_candidates:
                from_generator = self._generate(n_candidates - len(candidates))

                # Add the new data
                if self.decision_vars_seen is None:
                    n_existing_vars = 0
                    self.decision_vars_seen = self.vocs.variable_data(
                        from_generator
                    ).to_numpy()
                else:
                    n_existing_vars = self.decision_vars_seen.shape[0]
                    self.decision_vars_seen = np.concatenate(
                        (
                            self.decision_vars_seen,  # Must go first since first instance of unique elements are included
                            self.vocs.variable_data(
                                from_generator
                            ).to_numpy(),  # Do not accept repeated elements here
                        ),
                        axis=0,
                    )

                # Unique it and get the new candidates
                self.decision_vars_seen, idx = np.unique(
                    self.decision_vars_seen,
                    return_index=True,
                    axis=0,
                )
                n_removed += n_existing_vars + len(from_generator) - len(idx)
                idx = idx - n_existing_vars
                idx = idx[idx >= 0]
                for i in idx:
                    candidates.append(from_generator[i])
                self._logger.debug(
                    f"deduplicated generation round {round_idx} completed (n_removed={n_removed}, "
                    f"len(idx)={len(idx)}, n_existing_vars={n_existing_vars}, "
                    f"len(self.decision_vars_seen)={len(self.decision_vars_seen)})"
                )
                round_idx += 1

            # Hand candidates back to user
            candidates = candidates[:n_candidates]

        msg = f"generated {len(candidates)} candidates in {1000 * (time.perf_counter() - start_t):.2f}ms"
        if self.deduplicate_output:
            msg += f" (removed {n_removed} duplicate individuals)"
        self._logger.debug(msg)
        return candidates

    def _generate(self, n_candidates: int) -> list[dict]:
        """
        Generate candidate solutions without deduplication.

        This abstract method must be implemented by subclasses to provide
        the actual generation mechanism.

        Parameters
        ----------
        n_candidates : int
            Number of candidates to generate.

        Returns
        -------
        list of dict
            List of candidate solutions.
        """
        raise NotImplementedError
