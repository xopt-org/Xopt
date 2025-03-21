import numpy as np
from pydantic import Field, Discriminator
from typing import Dict, List, Optional, Annotated, Union
import pandas as pd
import os
from datetime import datetime
import logging
import time

from ...generator import StateOwner
from ..deduplicated import DeduplicatedGeneratorBase
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
def get_domination(pop_f: np.ndarray, pop_g: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute domination matrix for a population based on objective values and constraints. Determines domination
    relationships between all pairs of individuals in a population.

    Parameters
    ----------
    pop_f : numpy.ndarray
        Objective function values for each individual in the population,
        shape (n_individuals, n_objectives) where lower values are better.
    pop_g : numpy.ndarray, optional
        Constraint violation values for each individual, shape (n_individuals, n_constraints).
        Constraints are considered satisfied when <= 0.0. If None, unconstrained
        domination based solely on objectives is computed.

    Returns
    -------
    numpy.ndarray
        Boolean domination matrix of shape (n_individuals, n_individuals), where
        dom[i,j] = True means individual i dominates individual j.
    """
    # Compare all pairs of individuals based on domination
    dom = np.bitwise_and(
        (pop_f[:, None, :] <= pop_f[None, :, :]).all(axis=-1),
        (pop_f[:, None, :] < pop_f[None, :, :]).any(axis=-1),
    )

    if pop_g is not None:
        # If one individual is feasible and the other isn't, set domination
        feas = pop_g <= 0.0
        ind = np.bitwise_and(feas.all(axis=1)[:, None], ~feas.all(axis=1)[None, :])
        dom[ind] = True
        ind = np.bitwise_and(~feas.all(axis=1)[:, None], feas.all(axis=1)[None, :])
        dom[ind] = False

        # If both are infeasible, then the individual with the least constraint violation wins
        constraint_violation = np.sum(np.maximum(pop_g, 0), axis=1)
        comp = constraint_violation[:, None] < constraint_violation[None, :]
        ind = ~np.bitwise_or(feas.all(axis=1)[:, None], feas.all(axis=1)[None, :])
        dom[ind] = comp[ind]
    return dom


def fast_dominated_argsort_internal(dom: np.ndarray) -> List[np.ndarray]:
    """
    Used inside of `fast_dominated_argsort`. Call that function instead.

    Parameters
    ----------
    dom : np.ndarray
        The array of domination comparisons

    Returns
    -------
    list
        A list where each item is a set of the indices to the individuals contained in that domination rank
    """
    # Create the sets of dominated individuals, domination number, and first rank
    S = [np.nonzero(row)[0].tolist() for row in dom]
    N = np.sum(dom, axis=0)
    F = [np.where(N == 0)[0].tolist()]

    i = 0
    while len(F[-1]) > 0:
        Q = []
        for p in F[i]:
            for q in S[p]:
                N[q] -= 1
                if N[q] == 0:
                    Q.append(q)
        F.append(Q)
        i += 1

    # Remove last empty set
    F.pop()

    return F


def fast_dominated_argsort(
    pop_f: np.ndarray, pop_g: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """
    Performs a dominated sort on matrix of objective function values O.  This is a numpy implementation of the algorithm
    described in [1].

    A list of ranks is returned referencing each individual by its index in the objective matrix.

    References
    ----------
    [1] Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm:
        NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2). https://doi.org/10.1109/4235.996017

    Parameters
    ----------
    pop_f : np.ndarray
        (M, N) numpy array where N is the number of individuals and M is the number of objectives
    pop_g : np.ndarray, optional
        (M, N) numpy array where N is the number of individuals and M is the number of constraints, by default None

    Returns
    -------
    list
        List of ranks where each rank is a list of the indices to the individuals in that rank
    """
    dom = get_domination(pop_f, pop_g)
    return fast_dominated_argsort_internal(dom)


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
        dist[1:-1] += (Ps[2:] - Ps[:-2]) / (Ps[-1] - Ps[0] + 1e-300)

        # Unsort it
        unsort_ind = np.argsort(sort_ind)
        dist = dist[unsort_ind]

    return np.array(dist)


def crowded_comparison_argsort(pop_f: np.ndarray, pop_g: Optional[np.ndarray] = None):
    """
    Sorts the objective functions by domination rank and then by crowding distance (crowded comparison operator).
    Indices to individuals are returned in order of increasing value by crowded comparison operator.

    Parameters
    ----------
    pop_f : np.ndarray
        (M, N) numpy array where N is the number of individuals and M is the number of objectives
    pop_g : np.ndarray, optional
        The constraints, by default None

    Returns
    -------
    np.ndarray
        Numpy array of indices to sorted individuals
    """
    # Deal with NaNs
    pop_f = np.copy(pop_f)
    pop_f[~np.isfinite(pop_g)] = 1e300
    if pop_g is not None:
        pop_g = np.copy(pop_g)
        pop_g[~np.isfinite(pop_g)] = 1e300

    ranks = fast_dominated_argsort(pop_f, pop_g)
    inds = []
    for rank in ranks:
        dist = get_crowding_distance(pop_f[rank, :])
        inds.extend(np.array(rank)[np.argsort(dist)[::-1]])

    return np.array(inds)[::-1]


def get_fitness(pop_f: np.ndarray, pop_g: np.ndarray):
    """
    Get the "fitness" of each individual according to domination and crowding distance.

    Parameters
    ----------
    pop_f : np.ndarray
        The objectives
    pop_g : np.ndarray
        The constraints

    Returns
    -------
    np.ndarray
        The fitness of each individual
    """
    sort_ind = crowded_comparison_argsort(pop_f, pop_g)
    fitness = np.argsort(sort_ind)
    return fitness


def generate_child_binary_tournament(
    pop_x: np.ndarray,
    pop_f: np.ndarray,
    pop_g: np.ndarray,
    bounds: np.ndarray,
    mutate: MutationOperator,
    crossover: CrossoverOperator,
    fitness: Optional[np.ndarray] = None,
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
    pop_g : numpy.ndarray
        Constraint violation values of the population, shape (n_individuals, n_constraints).
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
    pop_x: np.ndarray, pop_f: np.ndarray, pop_g: np.ndarray, population_size: int
) -> np.ndarray:
    """
    Reduce population size by selecting the best individuals based on crowded comparison.

    Uses crowded comparison sorting to rank individuals in the population, then
    selects the top-ranked individuals to maintain the desired population size.

    Parameters
    ----------
    pop_x : numpy.ndarray
        Decision variables of the population, shape (n_individuals, n_variables).
    pop_f : numpy.ndarray
        Objective function values of the population, shape (n_individuals, n_objectives).
    pop_g : numpy.ndarray
        Constraint violation values of the population, shape (n_individuals, n_constraints).
    population_size : int
        Target size for the reduced population.

    Returns
    -------
    numpy.ndarray
        Indices of selected individuals, shape (population_size,).
    """
    inds = crowded_comparison_argsort(pop_f, pop_g)[::-1]
    inds = inds[:population_size]
    return inds


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

    population_size: int = Field(50, description="Population size")
    crossover_operator: Annotated[
        Union[
            SimulatedBinaryCrossover, DummyCrossover
        ],  # Dummy placeholder to keep discriminator code from failing
        Discriminator("name"),
    ] = SimulatedBinaryCrossover()
    mutation_operator: Annotated[
        Union[
            PolynomialMutation, DummyMutation
        ],  # Dummy placeholder to keep discriminator code from failing
        Discriminator("name"),
    ] = PolynomialMutation()

    # Output options
    output_dir: Optional[str] = None
    checkpoint_freq: int = Field(
        -1,
        description="How often (in generations) to save checkpoints (set to -1 to disable)",
    )
    log_level: int = Field(
        logging.INFO, description="Log message level output to log.txt"
    )
    _output_dir_setup: bool = (
        False  # Used in initializing the directory. PLEASE DO NOT CHANGE
    )
    _logger: Optional[logging.Logger] = None

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
    history_idx: List[List[int]] = Field(
        default=[],
        description="Xopt indices of the individuals in each population",
    )
    generation_start_t: float = Field(
        default_factory=time.perf_counter,
        description="When did the generation start, for logging",
        exclude=True,
    )

    # The population and returned children
    pop: List[Dict] = Field(default=[])
    child: List[Dict] = Field(default=[])

    def model_post_init(self, context):
        # Get a unique logger per object
        self._logger = logging.getLogger(f"{__name__}.NSGA2Generator.{id(self)}")
        self._logger.setLevel(self.log_level)

    def _generate(self, n_candidates: int) -> List[Dict]:
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
            pop_g = self.vocs.constraint_data(self.pop).to_numpy()
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

        # Pass to parent class for inclusion in self.data
        super().add_data(new_data)

        # Record the function evaluations
        self.fevals += len(new_data)
        self.child.extend(new_data.to_dict(orient="records"))
        self._logger.info(
            f"adding {len(new_data)} new evaluated individuals to generator"
        )

        round_idx = 0
        while len(self.child) >= self.population_size:
            self.pop.extend(self.child[: self.population_size])

            # Select using domination rank / crowding distance
            idx = cull_population(
                self.vocs.variable_data(self.pop).to_numpy(),
                self.vocs.objective_data(self.pop).to_numpy(),
                self.vocs.constraint_data(self.pop).to_numpy(),
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

                # Save all of the data
                self.data.to_csv(os.path.join(self.output_dir, "data.csv"), index=False)
                with open(os.path.join(self.output_dir, "vocs.txt"), "w") as f:
                    f.write(self.vocs.to_json())

                # Save this generation to the population file
                pop_df = pd.DataFrame(self.pop)
                pop_df["xopt_generation"] = self.n_generations
                csv_path = os.path.join(self.output_dir, "populations.csv")
                pop_df.to_csv(
                    csv_path, index=False, mode="a", header=not os.path.isfile(csv_path)
                )

                # Log some things
                self._logger.info(
                    f'saved optimization data to "{self.output_dir}" '
                    f"in {1000 * (time.perf_counter() - save_start_t):.2f}ms"
                )

                if self.checkpoint_freq > 0 and (
                    self.n_generations % self.checkpoint_freq == 0
                ):
                    # Create a base filename
                    os.makedirs(
                        os.path.join(self.output_dir, "checkpoints"), exist_ok=True
                    )
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
                    self._logger.info(f'saved checkpoint file "{checkpoint_path}"')

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
