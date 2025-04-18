# from xopt.generator import Generator
import numpy as np


def get_domination(pop_f: np.ndarray, pop_g: np.ndarray | None = None) -> np.ndarray:
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


def fast_dominated_argsort_internal(dom: np.ndarray) -> list[np.ndarray]:
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
    pop_f: np.ndarray, pop_g: np.ndarray | None = None
) -> list[np.ndarray]:
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


# def get_generator(name) -> Type[Generator]:
#     if name == "random":
#         from xopt.generators.random import RandomGenerator
#         return RandomGenerator
#     elif name == "upper_confidence_bound":
#         from xopt.generators.bayesian.upper_confidence_bound import \
#             UpperConfidenceBoundGenerator
#         return UpperConfidenceBoundGenerator
#     elif name == "mobo":
#         from xopt.generators.bayesian.mobo import MOBOGenerator
#         return MOBOGenerator
#     else:
#         raise ValueError(f"generator name {name} not found")
#
