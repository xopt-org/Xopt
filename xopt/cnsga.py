
"""
Continuous NSGA-II, NSGA-III

"""

from xopt import creator, vocs_tools, fitness_with_constraints

from xopt import creator, vocs_tools, fitness_with_constraints
from xopt.tools import full_path, random_settings_arrays, DummyExecutor, load_config, NpEncoder
from xopt import __version__
from xopt.vocs import VOCS

from deap import algorithms, base, tools

from tqdm.auto import tqdm

import numpy as np
import json

import logging
logger = logging.getLogger(__name__)

import warnings

from pprint import pprint
import time
import array
import random
import traceback
import os, sys





# Check for continuous integration
if 'CI' in os.environ:
    cnsga_logo = f"""
    Continuous Non-dominated Sorting Genetic Algorithm
    Version {__version__}
    """
else:
    cnsga_logo = f"""
    
    
     ▄████▄   ███▄    █   ██████   ▄████  ▄▄▄      
    ▒██▀ ▀█   ██ ▀█   █ ▒██    ▒  ██▒ ▀█▒▒████▄    
    ▒▓█    ▄ ▓██  ▀█ ██▒░ ▓██▄   ▒██░▄▄▄░▒██  ▀█▄  
    ▒▓▓▄ ▄██▒▓██▒  ▐▌██▒  ▒   ██▒░▓█  ██▓░██▄▄▄▄██ 
    ▒ ▓███▀ ░▒██░   ▓██░▒██████▒▒░▒▓███▀▒ ▓█   ▓██▒
    ░ ░▒ ▒  ░░ ▒░   ▒ ▒ ▒ ▒▓▒ ▒ ░ ░▒   ▒  ▒▒   ▓▒█░
      ░  ▒   ░ ░░   ░ ▒░░ ░▒  ░ ░  ░   ░   ▒   ▒▒ ░
    ░           ░   ░ ░ ░  ░  ░  ░ ░   ░   ░   ▒   
    ░ ░               ░       ░        ░       ░  ░
    ░                                              
    
    
    Continuous Non-dominated Sorting Genetic Algorithm
    Version {__version__}
    """




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
    
    var_labels = vocs_tools.skeys(var)
    obj_labels = vocs_tools.skeys(obj)
    
    bound_low = vocs_tools.var_mins(var)
    bound_up = vocs_tools.var_maxs(var)
    
    weights = vocs_tools.weight_list(obj)
    
    # Create MyFitness 
    if 'MyFitness' in dir(creator):
        del creator.MyFitness
    
    if n_con == 0:
        # Normal Fitness class
        creator.create('MyFitness', base.Fitness, weights=weights, labels=obj_labels)
    else:
        # Fitness with Constraints
        creator.create('MyFitness', fitness_with_constraints.FitnessWithConstraints, 
                   weights=weights, n_constraints=n_con, labels=obj_labels) 
    
    # Create Individual. Check if exists first. 
    if 'Individual' in dir(creator):
        del creator.Individual   
    creator.create('Individual', array.array, typecode='d', fitness=creator.MyFitness, 
                   labels=var_labels)    
    

      
    # Make toolbox
    toolbox = base.Toolbox()    
    
    # Register indivitual and population creation routines
    toolbox.register('attr_float', uniform, bound_low, bound_up)
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)        
    
    # Register mate and mutate functions
    toolbox.register('mate', tools.cxSimulatedBinaryBounded, low=bound_low, up=bound_up, eta=20.0)
    toolbox.register('mutate', tools.mutPolynomialBounded, low=bound_low, up=bound_up, eta=20.0, indpb=1.0/n_var)
    
    # Register NSGA selection algorithm.
    # NSGA-III should be better for 3 or more objectives
    if selection == 'auto':
        if len(obj) <= 2:
            selection = 'nsga2'
        else:
            selection='nsga3'

    if selection == 'nsga2':
        toolbox.register('select', tools.selNSGA2)
    
    # Doesn't work with constraints. TODO: investigate
    #elif selection == 'nsga2_log':
    #    toolbox.register('select', tools.selNSGA2, nd='log')        
    elif selection == 'nsga3':
        # Create uniform reference point
        ref_points = tools.uniform_reference_points(n_obj, 12)  
        toolbox.register('select', tools.selNSGA3, ref_points=ref_points)   
        
    elif selection == 'spea2':
        toolbox.register('select', tools.selSPEA2)
 
    else:
        raise ValueError(f'Invalid selection algorithm: {selection}')

        
    logger.info(f'Created toolbox with {n_var} variables, {n_con} constraints, and {n_obj} objectives.')
    logger.info(f'    Using selection algorithm: {selection}')
                         
    return toolbox
    
    
    
    
    
def cnsga_evaluate(vec, evaluate_f=None, vocs=None, include_inputs_and_outputs=True, verbose=True):
    """
    Evaluation function wrapper for use with cngsa. Returns dict with:
        'vec', 'obj', 'con', 'err'
    
    If a vocs is given, the function evaluate_f is assumed to have labeled inputs and outputs,
    and vocs will be used to form the output above. If include_inputs_and_outputs, then:
        'inputs', 'outputs'
    will be included in the returned dict. 
    
    Otherwise, evaluate_f should return pure numbers as:
        vec -> (objectives, constraints)

    This function will be evaulated by a worker. 
    
    Any exceptions will be caught, and this will return:
        error = True
        0 for all objectives
        -666.0 for all constraints

    """
    
    result = {}
    
    if vocs:
        # labeled inputs -> labeled outputs evaluate_f
        inputs = vocs_tools.inputs_from_vec(vec, vocs=vocs) 
    
    try:
    
        if vocs:
            
            # Evaluation
            inputs0 = inputs.copy()       # Make a copy, because the evaluate function might modify the inputs.
            outputs = evaluate_f(inputs0)
        
            obj_eval = vocs_tools.evaluate_objectives(vocs.objectives, outputs)
            con_eval = vocs_tools.evaluate_constraints(vocs.constraints, outputs)
            
        else:
        # Pure number function
            obj_eval, con_eval = evaluate_f(vec)
        
        err = False
    
    
    except Exception as ex:
        # No need to print a nasty logger exception
        logger.error(f'Exception caught in {__name__}')   
        outputs = {'Exception':  str(traceback.format_exc())}
        
        # Dummy output
        err = True
        obj_eval = [0.0]*len(vocs.objectives)
        con_eval = [-666.0]*len(vocs.constraints)
    
    finally:
         # Add these to result
        if include_inputs_and_outputs:
            result['inputs'] = inputs
            result['outputs'] = outputs
        
    
    result['vec'] = vec
    result['obj'] = obj_eval
    result['con'] = con_eval
    result['err'] = err
    
    return result







def pop_init(vocs, data):
    """
    Initialize a pop (list of creator.Indivituals) from vocs and keyed data.
    
    Data should have keys as in vocs.variables as arrays. 
    """

    # Get keys to look for in data
    varkeys = vocs_tools.skeys(vocs.variables)

    # extract vecs
    vecs = np.array([data[k] for k in varkeys]).T 
    
    # Check bounds
    for i, v in enumerate(varkeys):
        low, up = vocs.variables[v]
        assert vecs[:,i].min() >= low, 'Data below lower bound' 
        assert vecs[:,i].max() <= up,  'Data above upper bound' 
  
    # Pop must be a multiple of 4. Trim off any extras
    n_extra = len(vecs) % 4
    if n_extra > 0:
        logger.warning(f'Warning: trimming {n_extra} from initial population to make a multiple of 4.')
        vecs = vecs[:-n_extra]
   
    assert len(vecs) > 0, 'Population is empty'

    # Form individuals
    pop = []
    for vec in vecs:
        ind = creator.Individual(vec)
        pop.append(ind)
    
    return pop


def pop_init_random(vocs, n):
    """
    Returns a random population of size n. 
    """
    data = random_settings_arrays(vocs, n)
    return  pop_init(vocs, data)


def pop_to_data(vocs, pop, generation=0):
    """
    Pop should be a list of inds. 
    
    Returns a dict with
        'variables': dict with lists of input variable values
        'errors': corresponding error of these. 
        'vocs': the vocs used
        
    If inds had inputs and outputs, also returns:
        'inputs'
        'outputs'
    
    
    
    """
    
    data = {'variables':{}, 'generation':generation, 'vocs':vocs}

    vlist =  vocs_tools.skeys(vocs.variables) # get the correct order
    for i, v in enumerate(vlist):
        data['variables'][v] = [ind[i] for ind in pop]
    
    if not all([ind.fitness.valid for ind in pop]):
        return data
        
    #
    data['error'] = [ind.error for ind in pop]
    data['inputs'] = [ind.inputs for ind in pop]
    data['outputs'] = [ind.outputs for ind in pop]
                
    return data
    




# function to reform individual
def form_ind(res):
    vec, obj, con, err = res['vec'], res['obj'], res['con'], res['err']
    ind = creator.Individual(vec)
    
    ind.fitness.values = obj
    ind.fitness.cvalues = con
    ind.error = err
    
    if 'inputs' in res:
        ind.inputs = res['inputs']
    
    if 'outputs' in res:
        ind.outputs = res['outputs']
   
    return ind

# Only allow vectors to be sent to evaluate
def get_vec(ind):
    return array.array('d', [float(x) for x in ind])
def get_vecs(inds):
    return [get_vec(ind) for ind in inds]



#--------------------------------------------    
#--------------------------------------------

def cnsga(
          evaluate_f=None,
          vocs=None,
          executor=None,
          population=None,
          output_path=None,    
          max_generations = 2,
          population_size = 4,
          crossover_probability = 0.9,
          mutation_probability = 1.0,
          selection='auto',
          verbose=None, # deprecated
          toolbox=None,
          seed=None,    
          show_progress=False):
    """  
    Continuous NSGA-II constrained, multi-objective optimization algorithm.
    
    Futures method, uses an executor as defined in:
    https://www.python.org/dev/peps/pep-3148/
    
    Works with executors instantiated from:
       concurrent.futures.ProcessPoolExecutor
       concurrent.futures.ThreadPoolExecutor
       mpi4py.futures.MPIPoolExecutor
       dask.distributed.Client
       
    Requires either a DEAP toolbox or a vocs dict. 
    
    If an output_path is given, regular outputs:
        gen_{i}.json
        pop_{i}.json
    will be written for each generation i, and the best population at that generation.
    These files can be used for restarting the function. 
    
    
    Parameters
    ----------
    evaluate_f : callable
        Returns dict of outputs after problem has been evaluated
        
    vocs : dict
        Variables, objectives, constraints and statics dict.
        
    executor : Executor, default=None
        Executor object to run evaluate_f        

    population : dict or str, default=None
        Population dict or JSON filename to restart the algorithm from.
        If None, an initial population will be randomly generated.
        
    output_path : str, default=None
        Path to write output JSON files to. 

    population_size : int, default=4
        Population size. 
        Must be a multiple of 4

    max_generations : int, default = 2
        Maximum number of generations to advance. 

    n_initial_samples : int, default = 1
        Number of initial samples to take before using the model,
        overwritten by initial_x
        
    toolbox : deap.toolbox, default = None
        Optional toolbox from DEAP to use for further customizing the algorithm.
        Otherwise, vocs will be used to create the toolbox.
        
    seed: int, default=None
        random seed
        
    crossover_probability : float, default = 0.9
        Crossover probability
    
    mutation_probability : float, default = 1.0
        Mutation probability
    
    selection : str
        Selection algorith to use, one of:
            'auto' (default)
            'nsga2'
            'spea2'

    verbose: bool, default=None
        Deprecated, do not use.

    show_progress : bool, default=False
        If True, will show a progress bar for each generation

    Returns
    -------
    results : dict
        Dictionary with output data at the end of optimization
    
    
    """

    if vocs: vocs = VOCS.parse_obj(vocs)

    
    random.seed(seed)
    MU = population_size
    CXPB = float(crossover_probability)
    MUTPB = float(mutation_probability)
    
    
    if verbose is not None:
        warnings.warn('xopt.cnsga verbose option has been deprecated')                          
                         

    # Initial population

                    
    # Logo
    try:
        logger.info(cnsga_logo)
    except:
        logger.info('CNSGA')  # Windows has a problem with the logo
    
    if not executor:
        executor = DummyExecutor()
        logger.info('No executor given. Running in serial mode.')
    
    
    # Setup saving to file
    if output_path:
        path = full_path(output_path)
        assert os.path.exists(path), f'output_path does not exist {path}'
        
        def save(pop, prefix, generation):
            file = f'{prefix}{generation}.json'
            data = pop_to_data(vocs, pop, generation=generation)
            fullfile = os.path.join(path, file)
            with open(fullfile, 'w') as f:
                json.dump(data, f, ensure_ascii=True, cls=NpEncoder, indent=4)
            logger.info(f'Pop written to  {fullfile}')
        
    else:
        # Dummy save
        def save(pop, prefix, generation):
            pass
        
    # Toolbox
    if not toolbox:
        logger.info('Creating toolbox from vocs.')
        toolbox = cnsga_toolbox(vocs, selection=selection)
        toolbox.register('evaluate', cnsga_evaluate, evaluate_f=evaluate_f, vocs=vocs)
        
    # Initial pop
    if population:
        
        # Load JSON
        population = load_config(population)
        
        assert 'variables' in population, 'Population must have key: variables'
        
        pop = pop_init(vocs, population['variables'])
        
        if 'generation' in population:
            generation = population['generation']+1
            max_generations += generation
        else:
            generation=0
        MU = len(pop)
        logger.info(f'Initializing with existing population, size {MU}')
    else:
        generation = 0
        #pop = toolbox.population(n=MU)   
        pop = pop_init_random(vocs, n=MU)
        logger.info(f'Initializing with a new population, size {MU}')
    assert MU % 4 == 0, f'Population size (here {MU}) must be a multiple of 4'        
        
    logger.info(f'Maximum generations: {max_generations}')    
        
    
    # Individuals that need evaluating
    vecs = [get_vec(ind) for ind in pop if not ind.fitness.valid]

    # Initial population
    futures = [executor.submit(toolbox.evaluate, vec) for vec in vecs] 
    logger.info('____________________________________________________')
    logger.info(f'{MU} fitness calculations for initial generation')
    
    # Clear pop 
    pop = []
    for future in futures:
        res = future.result()
        ind = form_ind(res)
        pop.append(ind)
    logger.info('done.')
    logger.info('Submitting first batch of children')
    save(pop, 'initial_pop_', generation)
    
    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    
    # Make inital offspring to start the iteration
    vecs0 = get_vecs(algorithms.varAnd(pop, toolbox, CXPB, MUTPB) )

    # Submit evaluation of initial population
    futures = [executor.submit(toolbox.evaluate, vec) for vec in vecs0] 
    
    
    new_vecs = get_vecs(algorithms.varAnd(pop, toolbox, CXPB, MUTPB))
    new_offspring = []
       
    # Continuous loop
    t0 = time.time()
    done = False
    
    # Nice progress bar
    pbar = tqdm(total=len(futures), disable=not show_progress)
    
    while not done:
        # Check the status of all futures
        
       
        
        for ix in range(len(futures)):
         
            # Examine a future
            fut = futures[ix]
    
            if fut.done():
                res = fut.result()
                ind = form_ind(res)
                new_offspring.append(ind)  
                    
                # Increment the progress bar    
                pbar.update(1)

                # Refill inputs and save
                if len(new_vecs) == 0:
                    pbar.close()
                    
                    pbar = tqdm(total=(len(futures)), desc=f"Generation {generation}")                        
                    
                    t1 = time.time()
                    dt = t1-t0
                    t0 = t1
                    #logger.info('__________________________________________________________')
                    logger.info(f'Generation {generation} completed in {dt/60:0.5f} minutes')
                    generation += 1
                    
                    save(new_offspring, 'gen_', generation)
                    
                    pop = toolbox.select(pop + new_offspring, MU)
                    save(pop, 'pop_', generation)
                    
                    new_offspring = []
                    # New offspring
                    new_vecs = get_vecs(algorithms.varAnd(pop, toolbox, CXPB, MUTPB))                    
                    

                    if generation >= max_generations:
                        done = True
                            
                # Add new job for worker
                vec = new_vecs.pop()
                future = executor.submit(toolbox.evaluate, vec)
                futures[ix] = future        
                    

        
        # Slow down polling. Needed for MPI to work well. 
        time.sleep(0.001)
    
    # Close any progress bars
    pbar.update(len(futures))
#    pbar.clear()    
    pbar.close()
    
    # Cancel remaining jobs
    for future in futures:
        future.cancel()
    
    final_population = pop_to_data(vocs, pop, generation=generation)
            
    return final_population






