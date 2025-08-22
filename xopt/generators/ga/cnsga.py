import array
import logging
import os
import random
from typing import Dict, List, Optional

import pandas as pd
from deap import algorithms as deap_algorithms, base as deap_base, tools as deap_tools
from pydantic import ConfigDict, confloat, Field, PrivateAttr

import xopt.utils
from xopt.generator import Generator
from xopt.generators.ga import deap_creator
from xopt.generators.ga.deap_fitness_with_constraints import FitnessWithConstraints
from xopt.vocs import VOCS

logger = logging.getLogger(__name__)


class CNSGAGenerator(Generator):
    """
    Constrained Non-dominated Sorting Genetic Algorithm (CNSGA) generator.

    Attributes
    ----------
    name : str
        The name of the generator.
    supports_multi_objective : bool
        Indicates if the generator supports multi-objective optimization.
    population_size : int
        The population size for the genetic algorithm.
    crossover_probability : float
        The probability of crossover.
    mutation_probability : float
        The probability of mutation.
    population_file : Optional[str]
        The file path to load the population from (CSV format).
    output_path : Optional[str]
        The directory path to save the population files.
    _children : List[Dict]
        The list of children generated.
    _offspring : Optional[pd.DataFrame]
        The DataFrame containing the offspring.
    population : Optional[pd.DataFrame]
        The DataFrame containing the population.

    Methods
    -------
    create_children(self) -> List[Dict]
        Create children for the next generation.
    add_data(self, new_data: pd.DataFrame)
        Add new data to the generator.
    generate(self, n_candidates: int) -> List[Dict]
        Generate a specified number of candidate samples.
    write_offspring(self, filename: Optional[str] = None)
        Write the current offspring to a CSV file.
    write_population(self, filename: Optional[str] = None)
        Write the current population to a CSV file.
    load_population_csv(self, filename: str)
        Load a population from a CSV file.
    n_pop(self) -> int
        Convenience property for `population_size`.
    """

    name = "cnsga"
    supports_multi_objective: bool = True
    supports_constraints: bool = True
    supports_single_objective: bool = True
    population_size: int = Field(64, description="Population size")
    crossover_probability: confloat(ge=0, le=1) = Field(
        0.9, description="Crossover probability"
    )
    mutation_probability: confloat(ge=0, le=1) = Field(
        1.0, description="Mutation probability"
    )
    population_file: Optional[str] = Field(
        None, description="Population file to load (CSV format)"
    )
    output_path: Optional[str] = Field(
        None, description="Output path for population files"
    )
    _children: List[Dict] = PrivateAttr([])
    _offspring: Optional[pd.DataFrame] = PrivateAttr(None)
    population: Optional[pd.DataFrame] = Field(None)

    model_config = ConfigDict(extra="allow")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._loaded_population = (
            None  # use these to generate children until the first pop is made
        )

        # DEAP toolbox (internal)
        self._toolbox = cnsga_toolbox(self.vocs, selection="auto")

        if self.population_file is not None:
            self.load_population_csv(self.population_file)

        if self.output_path is not None:
            assert os.path.isdir(self.output_path), "Output directory does not exist"

    def create_children(self) -> List[Dict]:
        """
        Create children for the next generation.

        Returns
        -------
        List[Dict]
            A list of dictionaries containing the generated children.
        """
        # No population, so create random children
        if self.population is None:
            # Special case when pop is loaded from file
            if self._loaded_population is None:
                return self.vocs.random_inputs(self.n_pop, include_constants=False)
            else:
                pop = self._loaded_population
        else:
            pop = self.population

        # Use population to create children
        inputs = cnsga_variation(
            pop,
            self.vocs,
            self._toolbox,
            crossover_probability=self.crossover_probability,
            mutation_probability=self.mutation_probability,
        )
        return inputs.to_dict(orient="records")

    def add_data(self, new_data: pd.DataFrame):
        """
        Add new data to the generator.

        Parameters
        ----------
        new_data : pd.DataFrame
            The new data to be added.
        """
        if new_data is not None:
            self._offspring = pd.concat([self._offspring, new_data])

            # Next generation
            if len(self._offspring) >= self.n_pop:
                candidates = pd.concat([self.population, self._offspring])
                self.population = cnsga_select(
                    candidates, self.n_pop, self.vocs, self._toolbox
                )

                if self.output_path is not None:
                    self.write_offspring()
                    self.write_population()

                self._children = []  # reset children
                self._offspring = None  # reset offspring

    def generate(self, n_candidates: int) -> List[Dict]:
        """
        Generate a specified number of candidate samples.

        Parameters
        ----------
        n_candidates : int
            The number of candidate samples to generate.

        Returns
        -------
        List[Dict]
            A list of dictionaries containing the generated samples.
        """
        # Make sure we have enough children to fulfill the request
        while len(self._children) < n_candidates:
            self._children.extend(self.create_children())

        return [self._children.pop() for _ in range(n_candidates)]

    def write_offspring(self, filename: Optional[str] = None):
        """
        Write the current offspring to a CSV file.

        Parameters
        ----------
        filename : str, optional
            The file path to save the offspring. If None, a timestamped filename is generated.
        """
        if self._offspring is None:
            logger.warning("No offspring to write")
            return

        if filename is None:
            timestamp = xopt.utils.isotime(include_microseconds=True).replace(":", "_")
            filename = f"{self.name}_offspring_{timestamp}.csv"
            filename = os.path.join(self.output_path, filename)

        self._offspring.to_csv(filename, index_label="xopt_index")

    def write_population(self, filename: Optional[str] = None):
        """
        Write the current population to a CSV file.

        Parameters
        ----------
        filename : str, optional
            The file path to save the population. If None, a timestamped filename is generated.
        """
        if self.population is None:
            logger.warning("No population to write")
            return

        if filename is None:
            timestamp = xopt.utils.isotime(include_microseconds=True).replace(":", "_")
            filename = f"{self.name}_population_{timestamp}.csv"
            filename = os.path.join(self.output_path, filename)

        self.population.to_csv(filename, index_label="xopt_index")

    def load_population_csv(self, filename: str):
        """
        Load a population from a CSV file.

        Parameters
        ----------
        filename : str
            The file path to load the population from.
        """
        pop = pd.read_csv(filename, index_col="xopt_index")
        self._loaded_population = pop
        # This is a list of dicts
        self._children = self.vocs.convert_dataframe_to_inputs(
            pop[self.vocs.variable_names], include_constants=False
        ).to_dict(orient="records")
        logger.info(f"Loaded population of len {len(pop)} from file: {filename}")

    @property
    def n_pop(self) -> int:
        """
        Convenience property for `population_size`.

        Returns
        -------
        int
            The population size.
        """
        return self.population_size


def uniform(low: float, up: float, size: Optional[int] = None) -> List[float]:
    """
    Generate a list of uniform random numbers.

    Parameters
    ----------
    low : float
        The lower bound of the uniform distribution.
    up : float
        The upper bound of the uniform distribution.
    size : int, optional
        The number of random numbers to generate. If None, a single random number is generated.

    Returns
    -------
    List[float]
        A list of uniform random numbers.
    """
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


def cnsga_toolbox(vocs: VOCS, selection: str = "auto") -> deap_base.Toolbox:
    """
    Creates a DEAP toolbox from VOCS dict for use with CNSGA.

    Parameters
    ----------
    vocs : VOCS
        The VOCS object containing the variables, objectives, and constraints.
    selection : str, optional
        The selection algorithm to use. Options are "nsga2", "nsga3", "spea2", and "auto".
        Defaults to "auto".

    Returns
    -------
    deap_base.Toolbox
        The DEAP toolbox.

    Raises
    ------
    ValueError
        If an invalid selection algorithm is specified.
    """
    var, obj, con = vocs.variables, vocs.objectives, vocs.constraints
    n_var = len(var)
    n_obj = len(obj)
    n_con = len(con)

    var_labels = vocs.variable_names
    obj_labels = vocs.objective_names

    bound_low, bound_up = vocs.bounds
    # DEAP does not like arrays, needs tuples.
    bound_low = tuple(bound_low)
    bound_up = tuple(bound_up)

    # creator should assign already weighted values (for minimization)
    weights = tuple([-1] * n_obj)

    # Create MyFitness
    if "MyFitness" in dir(deap_creator):
        del deap_creator.MyFitness

    if n_con == 0:
        # Normal Fitness class
        deap_creator.create(
            "MyFitness", deap_base.Fitness, weights=weights, labels=obj_labels
        )
    else:
        # Fitness with Constraints
        deap_creator.create(
            "MyFitness",
            FitnessWithConstraints,
            weights=weights,
            n_constraints=n_con,
            labels=obj_labels,
        )

    # Create Individual. Check if exists first.
    if "Individual" in dir(deap_creator):
        del deap_creator.Individual
    deap_creator.create(
        "Individual",
        array.array,
        typecode="d",
        fitness=deap_creator.MyFitness,
        labels=var_labels,
    )

    # Make toolbox
    toolbox = deap_base.Toolbox()

    # Register individual and population creation routines
    # No longer needed
    # toolbox.register('attr_float', uniform, bound_low, bound_up)
    # toolbox.register('individual', deap_tools.initIterate, creator.Individual, toolbox.attr_float)
    # toolbox.register('population', deap_tools.initRepeat, list, toolbox.individual)

    # Register mate and mutate functions
    toolbox.register(
        "mate",
        deap_tools.cxSimulatedBinaryBounded,
        low=bound_low,
        up=bound_up,
        eta=20.0,
    )
    toolbox.register(
        "mutate",
        deap_tools.mutPolynomialBounded,
        low=bound_low,
        up=bound_up,
        eta=20.0,
        indpb=1.0 / n_var,
    )

    # Register NSGA selection algorithm.
    # NSGA-III should be better for 3 or more objectives
    if selection == "auto":
        selection = "nsga2"
        # TODO: fix this
        # if len(obj#) <= 2:
        #     select#ion = 'nsga2'
        # else# :
        #     selection='nsga3'

    if selection == "nsga2":
        toolbox.register("select", deap_tools.selNSGA2)

    elif selection == "spea2":
        toolbox.register("select", deap_tools.selSPEA2)

    else:
        raise ValueError(f"Invalid selection algorithm: {selection}")

    logger.info(
        f"Created toolbox with {n_var} variables, {n_con} constraints, and {n_obj} objectives."
    )
    logger.info(f"    Using selection algorithm: {selection}")

    return toolbox


def pop_from_data(data: pd.DataFrame, vocs: VOCS) -> List:
    """
    Return a list of DEAP deap_creator.Individual from a dataframe.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data.
    vocs : VOCS
        The VOCS object containing the variables, objectives, and constraints.

    Returns
    -------
    List[deap_creator.Individual]
        A list of DEAP individuals.
    """
    v = vocs.variable_data(data).to_numpy()
    o = vocs.objective_data(data).to_numpy()
    c = vocs.constraint_data(data).to_numpy()

    pop = list(map(deap_creator.Individual, v))
    for i, ind in enumerate(pop):
        ind.fitness.values = tuple(o[i, :])
        if c.size:
            ind.fitness.cvalues = tuple(c[i, :])

        ind.index = i

    return pop


def cnsga_select(
    data: pd.DataFrame, n: int, vocs: VOCS, toolbox: deap_base.Toolbox
) -> pd.DataFrame:
    """
    Applies DEAP's select algorithm to the population in data.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data.
    n : int
        The number of individuals to select.
    vocs : VOCS
        The VOCS object containing the variables, objectives, and constraints.
    toolbox : deap_base.Toolbox
        The DEAP toolbox.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the selected individuals.

    Note:
    -----
    This can be slow for large populations:
        NSGA2: Order(M N^2) for M objectives, N individuals
    """
    pop = pop_from_data(data, vocs)
    selected = toolbox.select(pop, n)  # Order(n^2)
    return data.iloc[[ind.index for ind in selected]]


def cnsga_variation(
    data: pd.DataFrame,
    vocs: VOCS,
    toolbox: deap_base.Toolbox,
    crossover_probability: float = 0.9,
    mutation_probability: float = 1.0,
) -> pd.DataFrame:
    """
    Varies the population (from variables in data) by applying crossover and mutation
    using DEAP's varAnd algorithm.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data.
    vocs : VOCS
        The VOCS object containing the variables, objectives, and constraints.
    toolbox : deap_base.Toolbox
        The DEAP toolbox.
    crossover_probability : float, optional
        The probability of crossover. Defaults to 0.9.
    mutation_probability : float, optional
        The probability of mutation. Defaults to 1.0.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the new individuals to evaluate.

    See:
    ----
    https://deap.readthedocs.io/en/master/api/algo.html#deap.algorithms.varAnd
    """
    v = vocs.variable_data(data).to_numpy()
    pop = list(map(deap_creator.Individual, v))

    children = deap_algorithms.varAnd(
        pop, toolbox, crossover_probability, mutation_probability
    )
    vecs = [[float(x) for x in child] for child in children]

    return vocs.convert_dataframe_to_inputs(
        pd.DataFrame(vecs, columns=vocs.variable_names), include_constants=False
    )
