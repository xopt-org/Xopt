from xopt import creator, vocs_tools, fitness_with_constraints
from xopt.tools import full_path, random_settings_arrays, DummyExecutor

from deap import algorithms, base, tools

import numpy as np
import json


from pprint import pprint
import time
import array
import random
import os, sys

"""
Continuous NSGA-II, NSGA-III

"""

cnsga_logo = """


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


Continous Non-dominated Sorting Genetic Algorithm

"""




def uniform(low, up, size=None):
    """
    
    """
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

    
    
def cnsga_toolbox(vocs, selection='auto', verbose=False):
    """
    Creates a DEAP toolbox from VOCS dict for use with cnsga. 
    
    """
    
    var, obj, con = vocs['variables'], vocs['objectives'], vocs['constraints']
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
        if verbose:
            print('Warning: Redefining creator.MyFitness')
    
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
        if verbose:
            print('Warning in cnsga_toolbox: Redefining creator.Individual')    
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
    elif selection == 'nsga3':
        toolbox.register('select', tools.selNSGA3)
    else:
        print('Error: invalid selection algorithm', selection)
        raise
    
    
    if verbose:
        print(f'Created toolbox with {n_var} variables, {n_con} constraints, and {n_obj} objectives.')
        print(f'    Using selection algorithm: {selection}')
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
    try:
            #f.write(str(vec))   
        if vocs:
            # labeled inputs -> labeled outputs evaluate_f
            inputs = vocs_tools.inputs_from_vec(vec, vocs=vocs)    
    
            # Evaluation
            outputs = evaluate_f(inputs)
        
            obj_eval = vocs_tools.evaluate_objectives(vocs['objectives'], outputs)
            con_eval = vocs_tools.evaluate_constraints(vocs['constraints'], outputs)
            
            # Add these to result
            if include_inputs_and_outputs:
                result['inputs'] = inputs
                result['outputs'] = outputs
        
        else:
        # Pure number function
            obj_eval, con_eval = evaluate_f(vec)
        
        err = False
    
    
    except Exception as ex:
        if verbose:
            print('Exception caught in cnsga_evaluate:', ex)

        # Dummy output
        err = True
        obj_eval = [0.0]*len(vocs['objectives'])
        con_eval = [-666.0]*len(vocs['constraints'])
    
    
    result['vec'] = vec
    result['obj'] = obj_eval
    result['con'] = con_eval
    result['err'] = err
    
    return result







def pop_init(vocs, data):
    """
    Initialize a pop (list of creator.Indivituals) from vocs and keyed data.
    
    Data should have keys as in vocs['variables'] as arrays. 
    """

    # Get keys to look for in data
    varkeys = vocs_tools.skeys(vocs['variables'])

    # extract vecs
    vecs = np.array([data[k] for k in varkeys]).T 
    
    # Check bounds
    for i, v in enumerate(varkeys):
        low, up = vocs['variables'][v]
        assert vecs[:,i].min() >= low, 'Data below lower bound' 
        assert vecs[:,i].max() <= up,  'Data above upper bound' 
  
    # Pop must be a multiple of 4. Trim off any extras
    n_extra = len(vecs) % 4
    if n_extra > 0:
        print(f'Warnning: trimming {n_extra} from initial population to make a multiple of 4.')
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

    vlist =  vocs_tools.skeys(vocs['variables']) # get the correct order
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
    
    if err:
        return ind
    
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

def cnsga(executor,
          vocs=None,
          population=None,
          toolbox=None,
          seed=None,
          evaluate_f=None,
          output_path=None,
          max_generations = 2,
          population_size = 4,
          crossover_probability = 0.9,
          mutation_probability = 1.0,
          selection='auto',
          verbose=True):
    """  
    Continuous NSGA-II, NSGA-III
    
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
    
    
    """

    
    random.seed(seed)
    MU = population_size
    CXPB = crossover_probability
    MUTPB = mutation_probability
    

    # Initial population

    # Verbose print helper
    def vprint(*a, **k):
        if verbose:
            print(*a, **k)
            sys.stdout.flush()
            
            
    
    vprint(cnsga_logo)
    
    if not executor:
        executor = DummyExecutor()
        vprint('No executor given. Running in serial mode.')
    
    
    # Setup saving to file
    if output_path:
        path = full_path(output_path)
        assert os.path.exists(path), f'output_path does not exist {path}'
        
        def save(pop, prefix, generation):
            file = f'{prefix}{generation}.json'
            data = pop_to_data(vocs, pop, generation=generation)
            fullfile = os.path.join(path, file)
            with open(fullfile, 'w') as f:
                json.dump(data, f, ensure_ascii=True, indent=4)
            vprint('Pop written to', fullfile)
        
    else:
        # Dummy save
        def save(pop, prefix, generation):
            pass
        
    # Toolbox
    if not toolbox:
        vprint('Creating toolbox from vocs.')
        toolbox = cnsga_toolbox(vocs, selection=selection, verbose=verbose)
        toolbox.register('evaluate', cnsga_evaluate, evaluate_f=evaluate_f, vocs=vocs, verbose=verbose)
        if verbose:
            print('vocs:')
            pprint(vocs) # Pretty print dict
        
    # Initial pop
    if population:
        pop = pop_init(vocs, population['variables'])
        if 'generation' in population:
            generation = population['generation']+1
        else:
            generation=0
        MU = len(pop)
        vprint(f'Initializing with existing population, size {MU}')
    else:
        generation = 0
        #pop = toolbox.population(n=MU)   
        pop = pop_init_random(vocs, n=MU)
        vprint(f'Initializing with a new population, size {MU}')
    assert MU % 4 == 0, f'Population size (here {MU}) must be a multiple of 4'        
        
    
    # Individuals that need evaluating
    vecs = [get_vec(ind) for ind in pop if not ind.fitness.valid]

    # Initial population
    futures = [executor.submit(toolbox.evaluate, vec) for vec in vecs] 
    vprint('____________________________________________________')
    vprint(f'{MU} fitness calculations for initial generation')
    
    # Clear pop 
    pop = []
    for future in futures:
        res = future.result()
        vprint('.', end='')
        ind = form_ind(res)
        pop.append(ind)
    vprint('done.')
    vprint('Submitting first batch of children')
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
    while not done:
        # Check the status of all futures
        for ix in range(len(futures)):
         
            # Examine a future
            fut = futures[ix]
    
            if fut.done():
                res = fut.result()
                ind = form_ind(res)
                new_offspring.append(ind)   
                vprint('.', end='')
                # Refill inputs and save
                if len(new_vecs) == 0:
                    t1 = time.time()
                    dt = t1-t0
                    t0 = t1
                    vprint('done.')
                    vprint('__________________________________________________________')
                    vprint(f'Generation {generation} completed in {dt/60:0.5f} minutes')
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
    
    # Cancel remaining jobs
    for future in futures:
        future.cancel()
    
    final_population = pop_to_data(vocs, pop, generation=generation)
            
    return final_population






