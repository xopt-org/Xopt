import numpy as np
from xopt.generator import Generator
from pydantic import Field, BaseModel, Discriminator
from typing import Dict, List, Optional, Literal, Annotated, Tuple, Union
import pandas as pd
import os
from datetime import datetime
import logging
import time


logger = logging.getLogger(__name__)


class MutationOperator(BaseModel):
    name: Literal["abstract"] = "abstract"

    def __call__(self, parent: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DummyMutation(BaseModel):
    name: Literal["_dummy"] = "_dummy"


class PolynomialMutation(MutationOperator):
    """
    Polynomial mutation operator for evolutionary algorithms.
    
    This operator performs mutation by adding a polynomial perturbation to the 
    parent solution, with the perturbation magnitude controlled by the distribution 
    parameter eta_m.
    
    Parameters
    ----------
    pm : float, optional
        Mutation probability for each decision variable, between 0 and 1.
        If None, defaults to 1/n where n is the number of variables.
    eta_m : int, default=20
        Mutation distribution parameter controlling the shape of the 
        perturbation. Larger values produce perturbations closer to the parent.
    """
    name: Literal["polynomial_mutation"] = "polynomial_mutation"
    pm: Annotated[Optional[float], Field(strict=True, ge=0, le=1, 
            description="Mutation probability or 1/n if None (n = # of vars)")] = None
    eta_m: Annotated[float, Field(strict=True, ge=0.0, description="Mutation distribution parameter")] = 20

    def __call__(self, parent: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """
        Apply polynomial mutation to a parent solution.
        
        Parameters
        ----------
        parent : numpy.ndarray
            Parent solution to be mutated, a 1D array of decision variables.
        bounds : numpy.ndarray
            Bounds for decision variables, shape (2, n) where n is the number
            of variables. bounds[0] contains lower bounds, bounds[1] contains
            upper bounds.
            
        Returns
        -------
        numpy.ndarray
            Mutated solution (child) with the same shape as the parent.
            
        Notes
        -----
        The mutation is applied with probability pm to each decision variable.
        The magnitude of the perturbation is controlled by eta_m, with larger
        values producing smaller perturbations. The mutation ensures that all
        variables remain within their bounds.
        """
        # Get the variables we are mutating
        if self.pm is None:
            pm = 1/parent.size
        else:
            pm = self.pm

        # The decision vars and bounds (only for variables we will mutate)
        do_mutation = np.random.random(parent.shape) < pm
        xm = parent[do_mutation]
        xl = bounds[0, do_mutation]
        xu = bounds[1, do_mutation]

        # Prepare for polynomial mutation
        mut_pow = 1.0 / (self.eta_m + 1.0)
        rand = np.random.random(xm.shape)
        deltaq = np.zeros(xm.shape)

        # Towards lower bound
        delta1 = (xm - xl) / (xu - xl)
        xy = 1.0 - delta1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (self.eta_m + 1.0)))
        dd = np.power(val, mut_pow) - 1.0
        deltaq[rand <= 0.5] = dd[rand <= 0.5]

        # Towards upper bound
        delta2 = (xu - xm) / (xu - xl)
        xy = 1.0 - delta2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (self.eta_m + 1.0)))
        dd = 1.0 - (np.power(val, mut_pow))
        deltaq[rand > 0.5] = dd[rand > 0.5]

        # Apply the mutation
        xm += deltaq * (xu - xl)
        
        # back in bounds if necessary (correct for floating point issues)
        xm[xm < xl] = xl[xm < xl]
        xm[xm > xu] = xu[xm > xu]

        # mutated values
        mutated = np.copy(parent)
        mutated[do_mutation] = xm

        return mutated


class CrossoverOperator(BaseModel):
    name: Literal["abstract"] = "abstract"

    def __call__(self, parent_a: np.ndarray, parent_b: np.ndarray, bounds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class DummyCrossover(BaseModel):
    name: Literal["_dummy"] = "_dummy"


class SimulatedBinaryCrossover(BaseModel):
    """
    Simulated Binary Crossover (SBX) operator for evolutionary algorithms.
    
    This crossover operator simulates the behavior of single-point crossover in
    binary-encoded genetic algorithms but is designed for real-valued variables.
    The method creates offspring that have a similar distance between them as their
    parents, with the spread controlled by the distribution parameter eta_c.
    
    Parameters
    ----------
    delta_1 : float, default=0.5
        Probability of crossing over each variable, between 0 and 1.
    delta_2 : float, default=0.5
        Probability of swapping crossed variables between children, between 0 and 1.
    eta_c : int, default=20
        Crossover distribution parameter that controls the spread of children
        solutions around parents. Larger values produce children closer to parents.
        Must be >= 0.
    
    References
    ----------
    [1] Deb, K., & Agrawal, R. B. (1995). Simulated binary crossover for continuous search space. Complex systems, 9(2), 115-148
    """
    name: Literal["simulated_binary_crossover"] = "simulated_binary_crossover"
    delta_1: Annotated[float, Field(strict=True, ge=0, le=1, description="Crossover probability")] = 0.5
    delta_2: Annotated[float, Field(strict=True, ge=0, le=1, description="Crossover probability")] = 0.5
    eta_c: Annotated[int, Field(strict=True, ge=0, description="Crossover distribution parameter")] = 20

    def __call__(self, parent_a: np.ndarray, parent_b: np.ndarray, bounds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply simulated binary crossover to generate two offspring from two parents.
        
        Parameters
        ----------
        parent_a : numpy.ndarray
            First parent solution, a 1D array of decision variables.
        parent_b : numpy.ndarray
            Second parent solution, a 1D array of decision variables with
            the same shape as parent_a.
        bounds : numpy.ndarray
            Bounds for decision variables, shape (2, n) where n is the number
            of variables. bounds[0] contains lower bounds, bounds[1] contains
            upper bounds.
            
        Returns
        -------
        tuple of numpy.ndarray
            A tuple containing two offspring solutions (child_a, child_b),
            each with the same shape as the parents.
            
        Notes
        -----
        The implementation follows these steps:
        1. Variables are crossed over with probability delta_1
        2. For crossed variables, SBX is applied with spread controlled by eta_c
        3. After crossover, variables may be swapped between children with probability delta_2
        4. Variables are clipped to lie in bounds correctly
        """
        # Get the number of variables
        n_var = parent_a.shape[0]

        # Create empty children
        c1 = np.empty(parent_a.shape[0])
        c2 = np.empty(parent_a.shape[0])

        # Get the indices which crossover and set those that did not
        crossed_over = np.random.random(n_var) < self.delta_1
        c1[~crossed_over] = parent_a[~crossed_over]
        c2[~crossed_over] = parent_b[~crossed_over]

        # Split the variables into those which match each other, those which are greater, and less than each other
        matching = np.isclose(parent_a, parent_b)
        less_than = np.bitwise_and(parent_a < parent_b, ~matching)
        greater_than = np.bitwise_and(parent_a > parent_b, ~matching)

        # Calculate the bounds scaling factor
        bl = np.ones(parent_a.shape[0])
        bu = np.ones(parent_a.shape[0])
        bl[less_than] = 1 + 2 * (parent_a[less_than] - bounds[0, less_than]) / (parent_b[less_than] - parent_a[less_than])
        bu[less_than] = 1 + 2 * (bounds[1, less_than] - parent_b[less_than]) / (parent_b[less_than] - parent_a[less_than])
        bl[greater_than] = 1 + 2 * (parent_b[greater_than] - bounds[0, greater_than]) / (parent_a[greater_than] - parent_b[greater_than])
        bu[greater_than] = 1 + 2 * (bounds[1, greater_than] - parent_a[greater_than]) / (parent_a[greater_than] - parent_b[greater_than])

        # Make the distribution symmetric (what Deb does)
        f = bl < bu
        bu[f] = bl[f]
        bl[~f] = bu[~f]

        # Raise everything to the distribution index power
        bl[~matching] = 1 - 1 / (2 * np.power(bl[~matching], 1 + self.eta_c))
        bu[~matching] = 1 - 1 / (2 * np.power(bu[~matching], 1 + self.eta_c))

        # Calculate the random scaling factor
        u = np.random.random(n_var) * (1.0 - 1e-6)
        b1 = u * bl
        b2 = u * bu

        # Convert to the distribution in Deb's paper
        f = b1 <= 0.5
        b1[f] = np.power(2 * b1[f], 1 / (self.eta_c + 1))
        b1[~f] = np.power(0.5 / (1 - b1[~f]), 1 / (self.eta_c + 1))
        f = b2 <= 0.5
        b2[f] = np.power(2 * b2[f], 1 / (self.eta_c + 1))
        b2[~f] = np.power(0.5 / (1 - b2[~f]), 1 / (self.eta_c + 1))

        # Perform the expansion
        f = np.bitwise_and(crossed_over, parent_a <= parent_b)
        c1[f] = (parent_a[f] + parent_b[f] + b1[f] * (parent_a[f] - parent_b[f])) / 2.0
        c2[f] = (parent_a[f] + parent_b[f] + b2[f] * (parent_b[f] - parent_a[f])) / 2.0
        f = np.bitwise_and(crossed_over, parent_a > parent_b)
        c1[f] = (parent_a[f] + parent_b[f] + b2[f] * (parent_a[f] - parent_b[f])) / 2.0
        c2[f] = (parent_a[f] + parent_b[f] + b1[f] * (parent_b[f] - parent_a[f])) / 2.0

        # Swap variables with probability delta_2
        f = np.bitwise_and(np.random.random(parent_a.shape[0]) < self.delta_2, crossed_over)
        t = c1[f]
        c1[f] = c2[f]
        c2[f] = t

        # Manually clip everything back into the boundaries
        f = c1 < bounds[0, :]
        c1[f] = bounds[0, f]
        f = c2 < bounds[0, :]
        c2[f] = bounds[0, f]
        f = c1 > bounds[1, :]
        c1[f] = bounds[1, f]
        f = c2 > bounds[1, :]
        c2[f] = bounds[1, f]

        # Return them
        return c1, c2


########################################################################################################################
# Helper functions
########################################################################################################################
def get_domination(pop_f, pop_g=None):
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


def fast_dominated_argsort_internal(dom):
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


def fast_dominated_argsort(pop_f, pop_g=None):
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


def get_crowding_distance(pop_f):
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


def crowded_comparison_argsort(pop_f, pop_g=None):
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


def get_fitness(pop_f, pop_g):
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


def generate_child_binary_tournament(pop_x, pop_f, pop_g, bounds, mutate: MutationOperator, crossover: CrossoverOperator):
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
        
    Returns
    -------
    numpy.ndarray
        The child solution with decision variables, shape (n_variables,).
    """
    # Perform selection
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


def cull_population(pop_x, pop_f, pop_g, population_size):
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

    def generate(self, n_candidates) -> list[dict]:
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
                    self.decision_vars_seen = self.vocs.variable_data(from_generator).to_numpy()
                else:
                    n_existing_vars = self.decision_vars_seen.shape[0]
                    self.decision_vars_seen = np.concatenate(
                        (
                            self.decision_vars_seen,  # Must go first since first instance of unique elements are included
                            self.vocs.variable_data(from_generator).to_numpy(),  # Do not accept repeated elements here
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
                logger.debug(f"deduplicated generation round {round_idx} completed (n_removed={n_removed}, len(idx)={len(idx)}, "
                             f"n_existing_vars={n_existing_vars}, len(self.decision_vars_seen)={len(self.decision_vars_seen)})")
                round_idx += 1

            # Hand candidates back to user
            candidates = candidates[:n_candidates]
        
        msg = f"generated {len(candidates)} candidates in {1000*(time.perf_counter() - start_t):.2f}ms"
        if self.deduplicate_output:
            msg += f" (removed {n_removed} duplicate individuals)"
        logger.info(msg)
        return candidates

    def _generate(self, n_candidates) -> list[dict]:
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


class NSGA2Generator(DeduplicatedGeneratorBase):
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
        XOpt indices of individuals in each generation.
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
        Union[SimulatedBinaryCrossover, DummyCrossover],  # Dummy placeholder to keep discriminator code from failing
        Discriminator("name")
    ] = SimulatedBinaryCrossover()
    mutation_operator: Annotated[
        Union[PolynomialMutation, DummyMutation],  # Dummy placeholder to keep discriminator code from failing
        Discriminator("name")
    ] = PolynomialMutation()
    
    # Output options
    output_dir: Optional[str] = None
    checkpoint_freq: int = 1
    _overwrite: bool = False  # Used in file overwrite protection logic. PLEASE DO NOT CHANGE
    
    # Metadata
    fevals: int = Field(0, description="Number of function evaluations the optimizer has seen up to this point")
    n_generations: int = Field(0, description="The number of generations completed so far")
    n_candidates: int = Field(0, description="The number of candidate solutions generated so far")
    history_idx: List[List[int]] = Field(default_factory=list, description="XOpt indices of the individuals in each population")
    generation_start_t: float = Field(default_factory=time.perf_counter,
                                      description="When did the generation start, for logging", exclude=True)

    # The population and returned children
    pop: List[Dict] = Field(default_factory=list)
    child: List[Dict] = Field(default_factory=list)

    def _generate(self, n_candidates: int) -> List[Dict]:
        start_t = time.perf_counter()
        
        # If we have a population create children, otherwise generate randomly sampled points
        if self.pop:
            # Get the variables
            var_names = sorted(self.vocs.variable_names)

            # Generate candidates one by one
            candidates = []
            for _ in range(n_candidates):
                candidates.append({k: v for k, v in zip(var_names, generate_child_binary_tournament(
                    self.vocs.variable_data(self.pop).to_numpy(), 
                    self.vocs.objective_data(self.pop).to_numpy(), 
                    self.vocs.constraint_data(self.pop).to_numpy(), 
                    self.vocs.bounds,
                    mutate=self.mutation_operator,
                    crossover=self.crossover_operator,
                ))})
            logger.debug(f"generated {n_candidates} candidates from generation {self.n_generations} "
                         f"in {1000*(time.perf_counter()-start_t):.2f}ms")
        else:
            vars = np.vstack([np.random.uniform(x[0], x[1], n_candidates) for x in self.vocs.bounds.T]).T
            candidates = [{k: v for k, v in zip(self.vocs.variable_names, individual)} for individual in vars]
            logger.debug(f"generated {n_candidates} random candidates in {1000*(time.perf_counter()-start_t):.2f}ms "
                         f"(no population exists yet)")
                
        # Add in useful tags for individuals
        for cand in candidates:
            # Record from which generation these candidates were generated from
            cand["xopt_parent_generation"] = self.n_generations
            
            # Record a unique index for every generated child
            cand["xopt_candidate_idx"] = self.n_candidates
            self.n_candidates += 1
        
        return candidates

    def add_data(self, new_data: pd.DataFrame):
        # Pass to parent class for inclusion in self.data
        super().add_data(new_data)

        # Record the function evaluations
        self.fevals += len(new_data)
        self.child.extend(new_data.to_dict(orient='records'))
        logger.info(f"adding {len(new_data)} new evaluated individuals to generator")

        round_idx = 0
        while len(self.child) >= self.population_size:
            self.pop.extend(self.child[:self.population_size])

            # Select using domination rank / crowding distance
            idx = cull_population(
                self.vocs.variable_data(self.pop).to_numpy(), 
                self.vocs.objective_data(self.pop).to_numpy(), 
                self.vocs.constraint_data(self.pop).to_numpy(), 
                self.population_size
            )
            self.pop = [self.pop[i] for i in idx]
            self.history_idx.append([x["xopt_candidate_idx"] for x in self.pop])

            # Get runtime information for the children used in this population
            rt = [x["xopt_runtime"] for x in self.child[:self.population_size]]
            perf_message = f"{np.mean(rt):.3f}s ({np.std(rt):.3f}s)"
            
            # Generate logging message
            n_feasible = np.sum((self.vocs.constraint_data(self.pop).to_numpy() <= 0.0).all(axis=1))
            n_err = np.sum([x["xopt_error"] for x in self.pop])
            logger.info(f"completed generation {self.n_generations+1} in {time.perf_counter()-self.generation_start_t:.3f}s"
                         f" (n_feasible={n_feasible}, n_err={n_err}, children_performance={perf_message}, "
                         f"add_data_round={round_idx}, fevals={self.fevals}, n_candidates={self.n_candidates})")
            round_idx += 1
            self.generation_start_t = time.perf_counter()
            
            # Reset children
            self.child = self.child[self.population_size:]
            self.n_generations += 1

            # Save the history file
            if self.output_dir is not None:
                save_start_t = time.perf_counter()

                # Check if directory exists and do collision avoidance
                if not self._overwrite:
                    counter = 2
                    output_dir_dedup = self.output_dir
                    while os.path.exists(output_dir_dedup):
                        output_dir_dedup = f"{self.output_dir}_{counter}"
                        counter += 1
                    logger.debug(f"detected existing output_dir \"{self.output_dir}\" and corrected "
                                 f"to \"{output_dir_dedup}\" to avoid overwriting")
                    self.output_dir = output_dir_dedup
                
                # Only avoid overwriting once per run
                self._overwrite = True
                
                # Setup the directory
                os.makedirs(self.output_dir, exist_ok=True)
                
                # Save all of the data
                self.data.to_csv(os.path.join(self.output_dir, "data.csv"))
                
                # Save this generation to the population file
                pop_df = pd.DataFrame(self.pop)
                pop_df["xopt_generation"] = self.n_generations
                csv_path = os.path.join(self.output_dir, f"populations.csv")
                pop_df.to_csv(csv_path, index=False, mode="a", header=not os.path.isfile(csv_path))
                
                # Log some things
                logger.info(f"saved optimization data to \"{self.output_dir}\" "
                            f"in {1000*(time.perf_counter()-save_start_t):.2f}ms")
                
                if self.n_generations % self.checkpoint_freq == 0:
                    # Create a base filename
                    os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
                    base_checkpoint_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
                    checkpoint_path = os.path.join(self.output_dir, "checkpoints", f"{base_checkpoint_filename}_1.txt")

                    # Check if file exists and increment counter until we find a free filename
                    counter = 2
                    while os.path.exists(checkpoint_path):
                        checkpoint_path = os.path.join(
                            self.output_dir, 
                            "checkpoints",
                            f"{base_checkpoint_filename}_{counter}.txt"
                        )
                        counter += 1

                    # Now we have a unique filename
                    with open(checkpoint_path, "w") as f:
                        f.write(self.to_json())
                    logger.debug(f"saved checkpoint file \"{checkpoint_path}\"")

    def __repr__(self) -> str:
        return (f"NSGA2Generator(pop_size={self.population_size}, "
                f"crossover={self.crossover_operator.__class__.__name__}, "
                f"mutation={self.mutation_operator.__class__.__name__}, "
                f"deduplicated={self.deduplicate_output}, "
                f"completed_gens={self.n_generations}, fevals={self.fevals})")
    
    def __str__(self) -> str:
        return self.__repr__()
    