from datetime import datetime
from itertools import chain
from pydantic import Field, Discriminator, model_validator
from typing import Annotated
import json
import logging
import numpy as np
import os
import pandas as pd
import time
import warnings

from ...errors import DataError
from ...generator import StateOwner
from ...vocs import VOCS
from ..deduplicated import DeduplicatedGeneratorBase
from ..utils import fast_dominated_argsort
from .operators import (
    PolynomialMutation,
    DummyMutation,
    SimulatedBinaryCrossover,
    DummyCrossover,
    MutationOperator,
    CrossoverOperator,
)


########################################################################################################################
# Helper functions
########################################################################################################################


def vocs_data_to_arr(data: list | np.ndarray) -> np.ndarray:
    """Force data coming from VOCS object into 2D numpy array (or None) for compatibility with helper functions"""
    if isinstance(data, list):
        data = np.ndarray(list)
    if data.size == 0:
        return None
    if len(data.shape) == 1:
        return data[:, None]
    if len(data.shape) == 2:
        return data
    raise ValueError(f"Unrecognized shape from VOCS data: {data.shape}")


def get_crowding_distance(pop_f: np.ndarray) -> np.ndarray:
    """
    Calculates NSGA-II style crowding distance as described in [1].

    References
    ----------
    [1] Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm:
        NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2). https://doi.org/10.1109/4235.996017

    Parameters
    ----------
    pop_f : np.ndarray
        (M, N) numpy array where N is the number of individuals and M is the number of objectives

    Returns
    -------
    np.ndarray
        Numpy array of crowding distance for each individual
    """
    dist = np.zeros(pop_f.shape[0])
    for m in range(pop_f.shape[1]):
        # Sort everything
        sort_ind = np.argsort(pop_f[:, m])
        dist = dist[sort_ind]
        Ps = pop_f[sort_ind, m]

        # Calculate distances for this objective
        dist[0] = np.inf
        dist[-1] = np.inf
        dist[1:-1] += (Ps[2:] - Ps[:-2]) / (
            Ps[-1] - Ps[0] + np.finfo(np.float64).smallest_normal
        )

        # Unsort it
        unsort_ind = np.argsort(sort_ind)
        dist = dist[unsort_ind]

    return np.array(dist)


def crowded_comparison_argsort(
    pop_f: np.ndarray, pop_g: np.ndarray | None = None
) -> np.ndarray:
    """
    Sorts the objective functions by domination rank and then by crowding distance (crowded comparison operator).
    Indices to individuals are returned in order of increasing value of fitness by crowded comparison operator.
    That is, the least fit individuals are returned first.

    Notes: NaN values are removed from the comparison and added back at the beginning (least fit direction) of
    the sorted indices.

    Parameters
    ----------
    pop_f : np.ndarray
        (N, M) numpy array where N is the number of individuals and M is the number of objectives
    pop_g : np.ndarray, optional
        The constraints, by default None

    Returns
    -------
    np.ndarray
        Numpy array of indices to sorted individuals
    """
    # Check for non-finite values in both pop_f and pop_g
    has_nan = np.any(~np.isfinite(pop_f), axis=1)
    if pop_g is not None:
        has_nan = has_nan | np.any(~np.isfinite(pop_g), axis=1)
    nan_indices = np.where(has_nan)[0]
    finite_indices = np.where(~has_nan)[0]

    # If all values are non-finite, return the original indices
    if len(finite_indices) == 0:
        return np.arange(pop_f.shape[0])

    # Extract only finite values for processing
    pop_f_finite = pop_f[finite_indices, :]

    # Handle constraints if provided
    pop_g_finite = None
    if pop_g is not None:
        pop_g_finite = pop_g[finite_indices, :]

    # Apply domination ranking
    ranks = fast_dominated_argsort(pop_f_finite, pop_g_finite)

    # Calculate crowding distance and sort within each rank
    sorted_finite_indices = []
    for rank in ranks:
        dist = get_crowding_distance(pop_f_finite[rank, :])
        sorted_rank = np.array(rank)[np.argsort(dist)[::-1]]
        sorted_finite_indices.extend(sorted_rank)

    # Map back to original indices and put nans at end
    sorted_original_indices = finite_indices[sorted_finite_indices]
    final_sorted_indices = np.concatenate([sorted_original_indices, nan_indices])

    return final_sorted_indices[::-1]


def get_fitness(pop_f: np.ndarray, pop_g: np.ndarray | None = None) -> np.ndarray:
    """
    Get the "fitness" of each individual according to domination and crowding distance.

    Parameters
    ----------
    pop_f : np.ndarray
        The objectives
    pop_g : np.ndarray / None
        The constraints, or None of no constraints

    Returns
    -------
    np.ndarray
        The fitness of each individual
    """
    return np.argsort(crowded_comparison_argsort(pop_f, pop_g))


def generate_child_binary_tournament(
    pop_x: np.ndarray,
    pop_f: np.ndarray,
    pop_g: np.ndarray | None,
    bounds: np.ndarray,
    mutate: MutationOperator,
    crossover: CrossoverOperator,
    fitness: np.ndarray | None = None,
) -> np.ndarray:
    """
    Creates a single child from the population using binary tournament selection, crossover, and mutation.

    Selection is performed using binary tournament where 4 random individuals are chosen
    and the best from each pair becomes a parent. The two parents undergo crossover
    to produce a child, which is then mutated before being returned.

    Parameters
    ----------
    pop_x : numpy.ndarray
        Decision variables of the population, shape (n_individuals, n_variables).
    pop_f : numpy.ndarray
        Objective function values of the population, shape (n_individuals, n_objectives).
    pop_g : numpy.ndarray / None
        Constraint violation values of the population, shape (n_individuals, n_constraints).
        None if no constraints.
    bounds : numpy.ndarray
        Bounds for decision variables, shape (2, n_variables) where bounds[0] are lower bounds
        and bounds[1] are upper bounds.
    mutate : MutationOperator
        Mutation operator to apply to the child solution.
    crossover : CrossoverOperator
        Crossover operator to apply to the parent solutions.
    fitness : np.ndarray
        The fitness of each individual (or None to compute from objectives and constraints)

    Returns
    -------
    numpy.ndarray
        The child solution with decision variables, shape (n_variables,).
    """
    # Perform selection
    if fitness is None:
        fitness = get_fitness(pop_f, pop_g)

    perm = np.random.permutation(pop_x.shape[0])
    if fitness[perm][0] > fitness[perm][1]:
        parent_1 = pop_x[perm, :][0, :]
    else:
        parent_1 = pop_x[perm, :][1, :]

    if fitness[perm][2] > fitness[perm][3]:
        parent_2 = pop_x[perm, :][2, :]
    else:
        parent_2 = pop_x[perm, :][3, :]

    # Run crossover
    child, _ = crossover(parent_1, parent_2, bounds)

    # Run mutation
    child = mutate(child, bounds)

    # Return the child
    return child


def cull_population(
    pop_x: np.ndarray, pop_f: np.ndarray, pop_g: np.ndarray | None, population_size: int
) -> np.ndarray:
    """
    Reduce population size by selecting the best individuals based on crowded comparison.

    Uses crowded comparison sorting to rank individuals in the population, then
    selects the top-ranked individuals to maintain the desired population size.

    Parameters
    ----------
    pop_x : numpy.ndarray
        Decision variables of the population, shape (n_individuals, n_variables).
    pop_f : numpy.ndarray / None
        Objective function values of the population, shape (n_individuals, n_objectives), None if no constraints.
    pop_g : numpy.ndarray
        Constraint violation values of the population, shape (n_individuals, n_constraints).
    population_size : int
        Target size for the reduced population.

    Returns
    -------
    numpy.ndarray
        Indices of selected individuals, shape (population_size,).
    """
    return crowded_comparison_argsort(pop_f, pop_g)[-population_size:]


########################################################################################################################
# Optimizer class
########################################################################################################################


class NSGA2Generator(DeduplicatedGeneratorBase, StateOwner):
    """
    Non-dominated Sorting Genetic Algorithm II (NSGA-II) generator.  Implements the NSGA-II algorithm
    for multi-objective optimization as described in [1]. This generator accomdates user selected mutation
    and crossover operators and performs selection with non-dominated sorting and crowding distance.

    References
    ----------
    [1] Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm:
        NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2). https://doi.org/10.1109/4235.996017

    Parameters
    ----------
    population_size : int, default=50
        Size of the population maintained across generations.
    crossover_operator : SimulatedBinaryCrossover or DummyCrossover, default=SimulatedBinaryCrossover()
        Operator used to perform crossover between parent solutions.
    mutation_operator : PolynomialMutation or DummyMutation, default=PolynomialMutation()
        Operator used to perform mutation on offspring solutions.
    output_dir : str, optional
        Directory to save algorithm state and population history.
    checkpoint_freq : int, default=1
        Frequency (in generations) at which to save checkpoints.
    checkpoint_file : str, optional
        Path to checkpoint file to load from. If provided, the generator will be initialized
        from the checkpoint state. The user-provided VOCS must match the checkpoint VOCS exactly.
        User-specified parameters will override checkpoint values.
    deduplicate_output : bool, default=True
        Whether to ensure all generated candidates are unique.

    Attributes
    ----------
    fevals : int
        Number of function evaluations performed so far.
    n_generations : int
        Number of generations completed.
    n_candidates : int
        Total number of candidate solutions generated.
    history_idx : list of list of int
        Xopt indices of individuals in each generation.
    pop : list of dict
        Current population of individuals.
    child : list of dict
        Buffer of evaluated offspring waiting to be incorporated into the population.

    Notes
    -----
    When `output_dir` is set to a path, the populations and all evaluated individuals will be written to the
    files "populations.csv" and "data.csv" respectively. Checkpoints are also saved every `checkpoint_freq` generation
    to a subdirectory. If the `output_dir` already exists at the first time output is created in the generator's lifetime,
    a number will be appended the output path to avoid overwriting previous data.

    The population file contains all of the populations with an index "xopt_generation" to indicate with which generation
    each row is associated.
    """

    name = "nsga2"
    supports_multi_objective: bool = True
    supports_constraints: bool = True
    supports_single_objective: bool = True

    # Checkpoint loading
    checkpoint_file: str | None = Field(
        None, description="Path to checkpoint file to load from", exclude=True
    )

    population_size: int = Field(50, description="Population size")
    crossover_operator: Annotated[
        (
            SimulatedBinaryCrossover | DummyCrossover
        ),  # Dummy placeholder to keep discriminator code from failing
        Discriminator("name"),
    ] = SimulatedBinaryCrossover()
    mutation_operator: Annotated[
        (
            PolynomialMutation | DummyMutation
        ),  # Dummy placeholder to keep discriminator code from failing
        Discriminator("name"),
    ] = PolynomialMutation()

    # Output options
    output_dir: str | None = None
    checkpoint_freq: int = Field(
        1,
        description="How often (in generations) to save checkpoints (set to -1 to disable)",
    )
    log_level: int = Field(
        logging.INFO, description="Log message level output to log.txt"
    )
    _output_dir_setup: bool = (
        False  # Used in initializing the directory. PLEASE DO NOT CHANGE
    )
    _logger: logging.Logger | None = None

    # Metadata
    fevals: int = Field(
        0,
        description="Number of function evaluations the optimizer has seen up to this point",
    )
    n_generations: int = Field(
        0, description="The number of generations completed so far"
    )
    n_candidates: int = Field(
        0, description="The number of candidate solutions generated so far"
    )
    history_idx: list[list[int]] = Field(
        default=[],
        description="Xopt indices of the individuals in each population",
    )
    generation_start_t: float = Field(
        default_factory=time.perf_counter,
        description="When did the generation start, for logging",
        exclude=True,
    )

    # The population and returned children
    pop: list[dict] = Field(default=[])
    child: list[dict] = Field(default=[])

    def model_post_init(self, context):
        # Get a unique logger per object
        self._logger = logging.getLogger(f"{__name__}.NSGA2Generator.{id(self)}")
        self._logger.setLevel(self.log_level)

    @staticmethod
    def _load_checkpoint_data(fname: str) -> dict:
        """
        Internal function to load generator data from checkpoint file as well as VOCS object.

        Parameters
        ----------
        fname : str
            Path to the checkpoint file

        Returns
        -------
        dict
            Dictionary containing VOCS and checkpoint data
        """
        # Load the VOCS object
        vocs_fname = os.path.join(os.path.dirname(fname), "../vocs.txt")
        if not os.path.exists(vocs_fname):
            raise ValueError(
                f'Could not load VOCS file at "{vocs_fname}". Complete NSGA2Generator '
                "output directory is required for loading from checkpoint."
            )
        with open(vocs_fname) as f:
            vocs = VOCS.from_dict(json.load(f))

        # Load the checkpoint
        with open(fname) as f:
            checkpoint_data = json.load(f)

        return {"vocs": vocs, **checkpoint_data}

    @model_validator(mode="before")
    @classmethod
    def load_from_checkpoint(cls, values):
        """
        Load from checkpoint file if checkpoint_file is provided.
        """
        # Case when a checkpoint file has been supplied
        if isinstance(values, dict) and "checkpoint_file" in values:
            checkpoint_file = values.pop("checkpoint_file")
            if checkpoint_file is not None:
                # Load checkpoint data
                checkpoint_data = cls._load_checkpoint_data(checkpoint_file)

                # Merge with user data precedence
                merged_data = {**checkpoint_data, **values}
                return merged_data

        # No checkpoint
        return values

    @model_validator(mode="after")
    def vocs_compatible(self):
        """
        Check that the VOCS object is compatible with our checkpoint
        For selection and the genetic operators to work correctly, all
        incoming variables, objectives, and constraints must exist as
        keys in pop/child
        """
        if self.pop or self.child:
            # The keys present in all individuals
            all_individuals = chain(self.pop, self.child)
            all_keys = set.intersection(*(set(x.keys()) for x in all_individuals))

            # Check that all required VOCS keys exist in the checkpoint populations
            if not all_keys.issuperset(
                self.vocs.variable_names
                + self.vocs.objective_names
                + self.vocs.constraint_names
            ):
                raise ValueError(
                    "User-provided VOCS is not compatible with existing population "
                    "or child data from checkpoint."
                )

            # Filter individuals outside of variable bounds
            # Use __setattr__ to not recursively apply validation
            n_ind = len(self.pop) + len(self.child)
            object.__setattr__(
                self, "pop", [x for x in self.pop if self.data_in_bounds(x)]
            )
            object.__setattr__(
                self, "child", [x for x in self.child if self.data_in_bounds(x)]
            )

            # Check how many individuals we filtered and report
            n_filtered = n_ind - (len(self.pop) + len(self.child))
            if n_filtered > 0:
                warnings.warn(
                    f"Filtered {n_filtered} individuals from population/children "
                    "that lay outside of variable bounds."
                )

        return self

    def data_in_bounds(self, data: dict) -> bool:
        """
        Returns true if every variable in the data dictionary is within bounds.
        """
        return all(
            bnd[0] <= data[key] <= bnd[1] for key, bnd in self.vocs.variables.items()
        )

    def _generate(self, n_candidates: int) -> list[dict]:
        self.ensure_output_dir_setup()
        start_t = time.perf_counter()

        # If we have a population create children, otherwise generate randomly sampled points
        if self.pop:
            # Get the variables
            var_names = sorted(self.vocs.variable_names)

            # Generate candidates one by one
            candidates = []
            pop_x = self.vocs.variable_data(self.pop).to_numpy()
            pop_f = self.vocs.objective_data(self.pop).to_numpy()
            pop_g = vocs_data_to_arr(self.vocs.constraint_data(self.pop).to_numpy())
            fitness = get_fitness(pop_f, pop_g)
            for _ in range(n_candidates):
                candidates.append(
                    {
                        k: v
                        for k, v in zip(
                            var_names,
                            generate_child_binary_tournament(
                                pop_x,
                                pop_f,
                                pop_g,
                                self.vocs.bounds,
                                mutate=self.mutation_operator,
                                crossover=self.crossover_operator,
                                fitness=fitness,
                            ),
                        )
                    }
                )
            self._logger.debug(
                f"generated {n_candidates} candidates from generation {self.n_generations} "
                f"in {1000 * (time.perf_counter() - start_t):.2f}ms"
            )
        else:
            vars = np.vstack(
                [
                    np.random.uniform(x[0], x[1], n_candidates)
                    for x in self.vocs.bounds.T
                ]
            ).T
            candidates = [
                {k: v for k, v in zip(self.vocs.variable_names, individual)}
                for individual in vars
            ]
            self._logger.debug(
                f"generated {n_candidates} random candidates in {1000 * (time.perf_counter() - start_t):.2f}ms "
                f"(no population exists yet)"
            )

        # Add in useful tags for individuals
        for cand in candidates:
            # Record from which generation these candidates were generated from
            cand["xopt_parent_generation"] = self.n_generations

            # Record a unique index for every generated child
            cand["xopt_candidate_idx"] = self.n_candidates
            self.n_candidates += 1

        return candidates

    def add_data(self, new_data: pd.DataFrame):
        self.ensure_output_dir_setup()

        # Validate data is at least compatible with selection / genetic operators
        vocs_names = (
            self.vocs.variable_names
            + self.vocs.objective_names
            + self.vocs.constraint_names
        )
        if not set(vocs_names).issubset(set(new_data.columns)):
            missing_cols = set(vocs_names).difference(set(new_data.columns))
            raise DataError(
                "New data must contain at least all variables, objectives, and constraints as columns"
                f" (missing columns: {missing_cols})"
            )

        # Pass to parent class for inclusion in self.data
        super().add_data(new_data)

        # Record the function evaluations
        self.fevals += len(new_data)
        self.child.extend(new_data.to_dict(orient="records"))
        self._logger.debug(
            f"adding {len(new_data)} new evaluated individuals to generator"
        )

        round_idx = 0
        while len(self.child) >= self.population_size:
            self.pop.extend(self.child[: self.population_size])

            # Select using domination rank / crowding distance
            idx = cull_population(
                self.vocs.variable_data(self.pop).to_numpy(),
                self.vocs.objective_data(self.pop).to_numpy(),
                vocs_data_to_arr(self.vocs.constraint_data(self.pop).to_numpy()),
                self.population_size,
            )
            self.pop = [self.pop[i] for i in idx]
            self.history_idx.append([x["xopt_candidate_idx"] for x in self.pop])

            # Get runtime information for the children used in this population
            rt = [x["xopt_runtime"] for x in self.child[: self.population_size]]
            perf_message = f"{np.mean(rt):.3f}s ({np.std(rt):.3f}s)"

            # Generate logging message
            n_feasible = np.sum(
                (self.vocs.constraint_data(self.pop).to_numpy() <= 0.0).all(axis=1)
            )
            n_err = np.sum([x["xopt_error"] for x in self.pop])
            self._logger.info(
                f"completed generation {self.n_generations + 1} in "
                f"{time.perf_counter() - self.generation_start_t:.3f}s"
                f" (n_feasible={n_feasible}, n_err={n_err}, children_performance={perf_message}, "
                f"add_data_round={round_idx}, fevals={self.fevals}, n_candidates={self.n_candidates})"
            )
            round_idx += 1
            self.generation_start_t = time.perf_counter()

            # Reset children
            self.child = self.child[self.population_size :]
            self.n_generations += 1

            # Save the history file
            if self.output_dir is not None:
                save_start_t = time.perf_counter()

                # Save all Xopt data
                self.data.to_csv(os.path.join(self.output_dir, "data.csv"), index=False)
                with open(os.path.join(self.output_dir, "vocs.txt"), "w") as f:
                    f.write(self.vocs.to_json())

                # Construct the DataFrame for this population
                pop_df = pd.DataFrame(self.pop)
                pop_df["xopt_generation"] = self.n_generations

                # Normalize the columns in the DataFrame
                # Avoid schema changing part way through optimization so we can write CSV in append mode
                columns = self.vocs.all_names + [
                    "xopt_generation",
                    "xopt_candidate_idx",
                    "xopt_runtime",
                    "xopt_error",
                ]
                pop_df = pop_df.reindex(columns=columns)

                # Write population DataFrame to file
                csv_path = os.path.join(self.output_dir, "populations.csv")
                pop_df.to_csv(
                    csv_path, index=False, mode="a", header=not os.path.isfile(csv_path)
                )

                # Log some things
                self._logger.debug(
                    f'saved optimization data to "{self.output_dir}" '
                    f"in {1000 * (time.perf_counter() - save_start_t):.2f}ms"
                )

                if self.checkpoint_freq > 0 and (
                    self.n_generations % self.checkpoint_freq == 0
                ):
                    self._save_checkpoint()

    def _save_checkpoint(self):
        # Confirm we are ready to save checkpoint
        if self.output_dir is None:
            raise ValueError("Cannot save checkpoint without an output directory")
        self.ensure_output_dir_setup()

        # Create a base filename
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        base_checkpoint_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            self.output_dir,
            "checkpoints",
            f"{base_checkpoint_filename}_1.txt",
        )

        # Check if file exists and increment counter until we find a free filename
        counter = 2
        while os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(
                self.output_dir,
                "checkpoints",
                f"{base_checkpoint_filename}_{counter}.txt",
            )
            counter += 1

        # Now we have a unique filename
        with open(checkpoint_path, "w") as f:
            f.write(self.to_json())
        self._logger.debug(f'saved checkpoint file "{checkpoint_path}"')

    def set_data(self, data):
        self.data = data

    def __repr__(self) -> str:
        return (
            f"NSGA2Generator(pop_size={self.population_size}, "
            f"crossover={self.crossover_operator.__class__.__name__}, "
            f"mutation={self.mutation_operator.__class__.__name__}, "
            f"deduplicated={self.deduplicate_output}, "
            f"completed_gens={self.n_generations}, fevals={self.fevals})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def ensure_output_dir_setup(self):
        if (self.output_dir is None) or self._output_dir_setup:
            return

        # Check if directory exists and do collision avoidance
        counter = 2
        output_dir_dedup = self.output_dir
        while os.path.exists(output_dir_dedup) and os.listdir(output_dir_dedup):
            output_dir_dedup = f"{self.output_dir}_{counter}"
            counter += 1
        self._logger.info(
            f'detected existing output_dir "{self.output_dir}" and corrected '
            f'to "{output_dir_dedup}" to avoid overwriting'
        )
        self.output_dir = output_dir_dedup

        # We are now setup
        self._output_dir_setup = True

        # Setup the directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up file logging
        log_file_path = os.path.join(self.output_dir, "log.txt")
        file_handler = logging.FileHandler(log_file_path, mode="w")
        file_handler.setLevel(self.log_level)

        # Use the same format as the default logger
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        self._logger.addHandler(file_handler)
        self._logger.info(f"routing log output to file: {log_file_path}")

    def close_log_file(self):
        """
        Closes out the log file (if used)
        """
        if self.output_dir is not None and self._output_dir_setup:
            # Remove all handlers from the logger
            for handler in list(self._logger.handlers):
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                self._logger.removeHandler(handler)
