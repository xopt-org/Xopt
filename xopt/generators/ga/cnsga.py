
from xopt.generators.ga import deap_creator
from xopt.generators.ga.deap_fitness_with_constraints import FitnessWithConstraints

from xopt import Generator

from deap import base as deap_base
from deap import tools as deap_tools
from deap import algorithms as deap_algorithms

import pandas as pd

import random
import array


from typing import List, Dict

import logging
logger = logging.getLogger(__name__)


from typing import List, Dict

class CNSGAGenerator(Generator):


    def __init__(self, vocs, *,  
    n_pop,
    data = None,
    crossover_probability = 0.9,
    mutation_probability = 1.0
    ): 

        self._vocs = vocs # TODO: use proper options
        self.n_pop = n_pop
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability


      # Internal data structures
        self.children = [] # unevaluated inputs. This should be a list of dicts.
        self.population = None # The latest population (fully evaluated)
        self.offspring  = None # Newly evaluated data, but not yet added to population

        # DEAP toolbox (internal)
        self.toolbox = cnsga_toolbox(vocs, selection='auto')

        if data is not None:
             self.population = cnsga_select(data, n_pop, vocs, self.toolbox)


    def create_children(self):

        # No population, so create random children
        if self.population is None:
            return [self.vocs.random_inputs() for _ in range(self.n_pop)]

        # Use population to create children
        inputs = cnsga_variation(self.population, self.vocs, self.toolbox,
            crossover_probability=self.crossover_probability, mutation_probability=self.mutation_probability)
        return inputs.to_dict(orient='records')


    def update_data(self, new_data: pd.DataFrame):
        self.offspring = pd.concat([self.offspring, new_data])

        # Next generation
        if len(self.offspring) >= self.n_pop:
            if self.population is None:   
                self.population = self.offspring.iloc[:self.n_pop]
                self.offspring = self.offspring.iloc[self.n_pop:]
            else:
                candidates = pd.concat([self.population, self.offspring])
                self.population = cnsga_select(candidates, self.n_pop, self.vocs, self.toolbox)
                self.children = [] # reset children
                self.offspring = None # reset offspring

    def generate(self, n_candidates) -> List[Dict]:
        """
        generate `n_candidates` candidates

        """

        # Make sure we have enough children to fulfill the request
        while len(self.children) < n_candidates:
            self.children.extend(self.create_children())
         
        return [self.children.pop() for _ in range(n_candidates)]






def uniform(low, up, size=None):
    """
    
    """
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

def cnsga_toolbox(vocs, selection='auto'):
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
    weights = tuple([-1]*n_obj)
    
    # Create MyFitness 
    if 'MyFitness' in dir(deap_creator):
        del deap_creator.MyFitness
    
    if n_con == 0:
        # Normal Fitness class
        deap_creator.create('MyFitness', deap_base.Fitness, weights=weights, labels=obj_labels)
    else:
        # Fitness with Constraints
        deap_creator.create('MyFitness', FitnessWithConstraints, 
                   weights=weights, n_constraints=n_con, labels=obj_labels) 
    
    # Create Individual. Check if exists first. 
    if 'Individual' in dir(deap_creator):
        del deap_creator.Individual   
    deap_creator.create('Individual', array.array, typecode='d', fitness=deap_creator.MyFitness, 
                   labels=var_labels)    
    

      
    # Make toolbox
    toolbox = deap_base.Toolbox()    
    
    # Register individual and population creation routines
    # No longer needed
    #toolbox.register('attr_float', uniform, bound_low, bound_up)
    #toolbox.register('individual', deap_tools.initIterate, creator.Individual, toolbox.attr_float)
    #toolbox.register('population', deap_tools.initRepeat, list, toolbox.individual)        
    
    # Register mate and mutate functions
    toolbox.register('mate', deap_tools.cxSimulatedBinaryBounded, low=bound_low, up=bound_up, eta=20.0)
    toolbox.register('mutate', deap_tools.mutPolynomialBounded, low=bound_low, up=bound_up, eta=20.0, indpb=1.0/n_var)
    
    # Register NSGA selection algorithm.
    # NSGA-III should be better for 3 or more objectives
    if selection == 'auto':
        selection = 'nsga2'
        # TODO: fix this
        # if len(obj#) <= 2:
        #     select#ion = 'nsga2'
        # else# :
        #     selection='nsga3'

    if selection == 'nsga2':
        toolbox.register('select', deap_tools.selNSGA2)
    
    elif selection == 'spea2':
        toolbox.register('select', deap_tools.selSPEA2)
 
    else:
        raise ValueError(f'Invalid selection algorithm: {selection}')

        
    logger.info(f'Created toolbox with {n_var} variables, {n_con} constraints, and {n_obj} objectives.')
    logger.info(f'    Using selection algorithm: {selection}')
                         
    return toolbox




def pop_from_data(data, vocs):
    """
    Return a list of DEAP deap_creator.Individual from a dataframe
    """
    ix = data.index.to_numpy()
    v = vocs.variable_data(data).to_numpy()
    o = vocs.objective_data(data).to_numpy()
    c = vocs.constraint_data(data).to_numpy()

    pop = list(map(deap_creator.Individual, v))
    for i, ind in enumerate(pop):
        ind.fitness.values = tuple(o[i, :])
        ind.fitness.cvalues = tuple(c[i, :])
        ind.index = ix[i]
 
    return pop

def cnsga_select(data, n, vocs, toolbox):
    """
    Applies DEAP's select algorithm to the population in data.

    Note that this can be slow for large populations:
        NSGA2: Order(M N^2) for M objectives, N individuals
    """
    pop = pop_from_data(data, vocs)
    selected = toolbox.select(pop, n) # Order(n^2)
    return data.loc[ [ind.index for ind in selected] ]


def cnsga_variation(data, vocs, toolbox, crossover_probability=0.9, mutation_probability=1.0):
    """
    Varies the population (from variables in data) by applying crossover and mutation
    using DEAP's varAnd algorithm. 

    Returns an input dataframe with the new individuals to evaluate.

    See: https://deap.readthedocs.io/en/master/api/algo.html#deap.algorithms.varAnd
    """
    v = vocs.variable_data(data).to_numpy()
    pop = list(map(deap_creator.Individual, v))

    children = deap_algorithms.varAnd(pop, toolbox, crossover_probability, mutation_probability)
    vecs = [[float(x) for x in child] for child in children]  

    return vocs.convert_dataframe_to_inputs(pd.DataFrame(vecs, columns=vocs.variable_names ))


