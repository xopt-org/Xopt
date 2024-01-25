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

# xopt.utils.isotime() # works
# from xopt.utils import isotime # circular import


logger = logging.getLogger(__name__)


class CNSGAGenerator(Generator):
    name = "cnsga"
    supports_multi_objective: bool = True
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
        None, description="Output path for population " "files"
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

        # if data is not None:
        #    self.population = cnsga_select(data, n_pop, vocs, self.toolbox)

    def create_children(self) -> List[Dict]:
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

    def generate(self, n_candidates) -> list[dict]:
        """
        generate `n_candidates` candidates

        """

        # Make sure we have enough children to fulfill the request
        while len(self._children) < n_candidates:
            self._children.extend(self.create_children())

        return [self._children.pop() for _ in range(n_candidates)]

    def write_offspring(self, filename=None):
        """
        Write the current offspring to a CSV file.

        Similar to write_population
        """
        if self._offspring is None:
            logger.warning("No offspring to write")
            return

        if filename is None:
            filename = f"{self.name}_offspring_{xopt.utils.isotime(include_microseconds=True)}.csv"
            filename = os.path.join(self.output_path, filename)

        self._offspring.to_csv(filename, index_label="xopt_index")

    def write_population(self, filename=None):
        """
        Write the current population to a CSV file.

        Similar to write_offspring
        """
        if self.population is None:
            logger.warning("No population to write")
            return

        if filename is None:
            filename = f"{self.name}_population_{xopt.utils.isotime(include_microseconds=True)}.csv"
            filename = os.path.join(self.output_path, filename)

        self.population.to_csv(filename, index_label="xopt_index")

    def load_population_csv(self, filename):
        """
        Read a population from a CSV file.
        These will be reverted back to children for re-evaluation.
        """
        pop = pd.read_csv(filename, index_col="xopt_index")
        self._loaded_population = pop
        # This is a list of dicts
        self._children = self.vocs.convert_dataframe_to_inputs(
            pop[self.vocs.variable_names], include_constants=False
        ).to_dict(orient="records")
        logger.info(f"Loaded population of len {len(pop)} from file: {filename}")

    @property
    def n_pop(self):
        """
        Convenience name for `options.population_size`
        """
        return self.population_size


def uniform(low, up, size=None):
    """ """
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


def cnsga_toolbox(vocs, selection="auto"):
    """
    Creates a DEAP toolbox from VOCS dict for use with cnsga.

    Selection options:

    nsga2: Standard NSGA2 [Deb2002] selection
    nsga3: NSGA3 [Deb2014] selection
    spea2: SPEA-II [Zitzler2001] selection
    auto: will choose nsga2 for <= 2 objectives, otherwise nsga3


    See DEAP code for details.

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


def pop_from_data(data, vocs):
    """
    Return a list of DEAP deap_creator.Individual from a dataframe
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


def cnsga_select(data, n, vocs, toolbox):
    """
    Applies DEAP's select algorithm to the population in data.

    Note that this can be slow for large populations:
        NSGA2: Order(M N^2) for M objectives, N individuals
    """
    pop = pop_from_data(data, vocs)
    selected = toolbox.select(pop, n)  # Order(n^2)
    return data.iloc[[ind.index for ind in selected]]


def cnsga_variation(
    data, vocs, toolbox, crossover_probability=0.9, mutation_probability=1.0
):
    """
    Varies the population (from variables in data) by applying crossover and mutation
    using DEAP's varAnd algorithm.

    Returns an input dataframe with the new individuals to evaluate.

    See: https://deap.readthedocs.io/en/master/api/algo.html#deap.algorithms.varAnd
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
