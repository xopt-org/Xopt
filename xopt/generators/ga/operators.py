import numpy as np
from pydantic import Field, field_validator
from typing import Optional, Literal, Annotated, Tuple

from ...pydantic import XoptBaseModel


class MutationOperator(XoptBaseModel):
    name: Literal["abstract"] = "abstract"

    @field_validator("name", mode="after")
    def validate_files(cls, value, info):
        """
        Hack to override the wildcard before validator in `XoptBaseModel` for
        the discriminator field. Before validators are dissallowed in this case.
        """
        return value

    def __call__(self, parent: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DummyMutation(MutationOperator):
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
    pm: Annotated[
        Optional[float],
        Field(
            strict=True,
            ge=0,
            le=1,
            description="Mutation probability or 1/n if None (n = # of vars)",
        ),
    ] = None
    eta_m: Annotated[
        float, Field(strict=True, ge=0.0, description="Mutation distribution parameter")
    ] = 20

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
            pm = 1 / parent.size
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
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (
            np.power(xy, (self.eta_m + 1.0))
        )
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


class CrossoverOperator(XoptBaseModel):
    name: Literal["abstract"] = "abstract"

    @field_validator("name", mode="after")
    def validate_files(cls, value, info):
        """
        Hack to override the wildcard before validator in `XoptBaseModel` for
        the discriminator field. Before validators are dissallowed in this case.
        """
        return value

    def __call__(
        self, parent_a: np.ndarray, parent_b: np.ndarray, bounds: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class DummyCrossover(CrossoverOperator):
    name: Literal["_dummy"] = "_dummy"


class SimulatedBinaryCrossover(CrossoverOperator):
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
    delta_1: Annotated[
        float, Field(strict=True, ge=0, le=1, description="Crossover probability")
    ] = 0.5
    delta_2: Annotated[
        float, Field(strict=True, ge=0, le=1, description="Crossover probability")
    ] = 0.5
    eta_c: Annotated[
        int, Field(strict=True, ge=0, description="Crossover distribution parameter")
    ] = 20

    def __call__(
        self, parent_a: np.ndarray, parent_b: np.ndarray, bounds: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        bl[less_than] = 1 + 2 * (parent_a[less_than] - bounds[0, less_than]) / (
            parent_b[less_than] - parent_a[less_than]
        )
        bu[less_than] = 1 + 2 * (bounds[1, less_than] - parent_b[less_than]) / (
            parent_b[less_than] - parent_a[less_than]
        )
        bl[greater_than] = 1 + 2 * (
            parent_b[greater_than] - bounds[0, greater_than]
        ) / (parent_a[greater_than] - parent_b[greater_than])
        bu[greater_than] = 1 + 2 * (
            bounds[1, greater_than] - parent_a[greater_than]
        ) / (parent_a[greater_than] - parent_b[greater_than])

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
        f = np.bitwise_and(
            np.random.random(parent_a.shape[0]) < self.delta_2, crossed_over
        )
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
