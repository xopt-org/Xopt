#!/usr/bin/env python

# Adapted from DEAP's NSGA-II example


import array
import random
import time



import pickle # for checkpointing. Json doesn't work on the array object

import os
import sys

import numpy as np

#from scoop import ### logger

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence
from deap import creator
from deap import tools
from xdeap import fitness_with_constraints


def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

#-------------------------------
# Population info

def write_checkpoint(filename, population=None, generation=0, logbook=None):
    if not logbook:
        logbook = tools.Logbook()
    if not population:
        print('ERROR, must have a population!')
    cp = dict(population=population, generation=generation, 
              logbook=logbook, rndstate=random.getstate())
    pickle.dump(cp, open(filename, 'wb'))

def numeric_keys(info):
    x = []
    for key in info:
        if (isinstance(info[key], int) or isinstance(info[key], float)):
            x.append(key)
    return x

def write_txt_population(population, filename):

    f = open(filename, 'w')
    
    # Get info headers
    reference_keys = numeric_keys(population[0].fitness.info)
    for ind in population:
        new_keys =  numeric_keys(ind.fitness.info)
        if len(new_keys) > len(reference_keys): 
            reference_keys = new_keys
    reference_keys.sort()
    
    # header (check for backwards compatibility - hasattr can later be removed)
    if hasattr(population[0], 'objective_labels'):
      header = population[0].objective_labels + ['constraints_satisfied'] + population[0].variable_labels + reference_keys          
      for x in header:
        f.write(x)
        f.write(' ')
      f.write('\n')
    
    
    for x in population:   
        # objectives
        for val in x.fitness.values:
            f.write(str(val))
            f.write(' ')
        # Constraints
        feasible = x.fitness.feasible()
        f.write(str(feasible))
        f.write(' ')
        # vector
        for var in x: 
            f.write(str(var))
            f.write(' ') 

        info = x.fitness.info
        for key in reference_keys:
            if key in info:
                f.write(str(info[key]))
            else:
                f.write('None')
            f.write(' ') 
 
        f.write('\n')
    f.close()  


def read_txt_population(population, filename):
    '''
    Reads txt population with header.
    '''
    var_labels = population[0].variable_labels
  
    # Get header and data
    f = open(filename, 'r')
    dat = []
    header = f.readline().split()
    for line in f:
        s = line.split()
        if len(s) > 0:
            dat.append(s)
    f.close()

    n_to_set = min(len(dat), len(population))

   # identify variables for replacement and replace
    for ix_label in range(len(header)):
        label = header[ix_label]

        ix = [i for i,x in enumerate(var_labels) if x == label]
        if len(ix) == 1:
            ix = ix[0]
            for i in range(n_to_set):
                population[i][ix] = float(dat[i][ix_label])
        elif len(ix) == 0:
            print('Warning: no variable found for: '+label)
        else: 
            print('Error: Multiple ('+str(len(ix))+') variable labels for: '+label)
            print('       No replacements performed')
    

def upgrade_pkl_population(oldfile, newfile, variable_labels, objective_labels ):
    '''
    Adds .variable_labels and .objective_labels to every individual in the population
    '''
    print('Upgrading '+oldfile)
    cp = pickle.load(open(oldfile, 'rb') )
    population = cp['population']
    for ind in population:
      ind.variable_labels = variable_labels 
      ind.objective_labels = objective_labels
    pickle.dump(cp, open(newfile, 'wb') )
    print('  '+newfile+' written')
    
# def read_txt_population(filename, population, dir=OUTPUT_DIR):
#     f = open(dir+filename, 'r')
#     ix = 0
#     for line in f:
#         vec = [float(x) for x in line.split()[3:] ]
#         #print(vec)
#         #print(population[ix])
#         population[ix][0:] = array.array('d', vec[0:])
#         #print(population[ix].fitness.valid)
#         ix = ix+1
#     print('reloaded population: ', filename)
#     f.close()
  
#-------------------------------

def nsga2_toolbox(N_DIM=1, WEIGHTS = (1), N_CONSTRAINTS = 0, BOUND_LOW = [0.0], BOUND_UP=[1.0], VARIABLE_LABELS=['variable_1'], OBJECTIVE_LABELS=['objective_1']): 
    '''
    Returns a DEAP toolbox for use in a NSGA2 loop
    '''
    creator.create('MyFitness', fitness_with_constraints.FitnessWithConstraints, weights=WEIGHTS, n_constraints = N_CONSTRAINTS)
    creator.create('Individual', array.array, typecode='d', fitness=creator.MyFitness, variable_labels=VARIABLE_LABELS, objective_labels=OBJECTIVE_LABELS)
    # Don't use Numpy version. See advanced tutorial for why. 
    #creator.create('Individual', np.ndarray,  fitness=creator.MyFitness, variable_labels=VARIABLE_LABELS, objective_labels=OBJECTIVE_LABELS)


    toolbox = base.Toolbox()
    
    
    
    toolbox.register('attr_float', uniform, BOUND_LOW, BOUND_UP, N_DIM)
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register('mate', tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    toolbox.register('mutate', tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/N_DIM)
    toolbox.register('select', tools.selNSGA2)

    return toolbox

# NSGA2 config
def params_from_config(nsga2_config):
    '''
    config: ConfigParser()
    
    Returns kwarg dict for main NSGA2 loop
    '''
    nsga2_params = {}
    outdir = nsga2_config.get('output_dir')
    if outdir == '.':
        outdir = os.getcwd()    
    nsga2_params['output_dir'] = outdir
    for x in ['max_generations', 'population_size', 'checkpoint_frequency']:
        nsga2_params[x] = int(nsga2_config[x])
    checkpoint = nsga2_config['checkpoint']
    if checkpoint == '':
        checkpoint = None
    else:
        if not os.path.exists(checkpoint):
            sys.exit("ERROR: Checkpoint file doens't exist: "+checkpoint)
    nsga2_params['checkpoint'] = checkpoint
    nsga2_params['do_history'] = nsga2_config.getboolean('do_history')
    
    nsga2_params['abort_file'] = nsga2_config.get('abort_file')
    return nsga2_params
    


def main(toolbox, output_dir = '', checkpoint=None, seed=None,
         max_generations = 2, population_size = 4, checkpoint_frequency = 1,
         crossover_probability = 0.9, do_history=False, abort_file='', 
         skip_checkpoint_eval=False):
    '''
    Main loop for a NSGA-II optimization
    
    returns final population and logbook
    '''
    random.seed(seed)
    
    # History
    if do_history:
        history = tools.History()
        toolbox.decorate('mate', history.decorator)
        toolbox.decorate('mutate', history.decorator)

    NGEN = max_generations
    MU =  population_size  
    CXPB = crossover_probability

    assert ( MU % 4 == 0) # Must be multiple of 4
    
    CHECKPOINT_FREQUENCY = checkpoint_frequency

    # Init statistics 
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean, axis=0)
    stats.register('std', np.std, axis=0)
    stats.register('min', np.min, axis=0)
    stats.register('max', np.max, axis=0)

    
    #  LOAD CHECKPOINT
    if checkpoint:
        # A file name has been given, then load the data from the file
        cp = pickle.load(open(checkpoint, 'rb'))
        population = cp['population']
        start_gen = cp['generation'] + 1
        #halloffame = cp['halloffame']
        logbook = cp['logbook']
        random.setstate(cp['rndstate'])
        ### logger.info('Loaded checkpoint: '+checkpoint)
    else:
        # Start a new evolution
        population = toolbox.population(n=MU)
        start_gen = 0
        #halloffame = tools.HallOfFame(maxsize=1)
        logbook = tools.Logbook()
        logbook.header = 'gen', 'evals', 'std', 'min', 'avg', 'max'
     
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population] #Reevaluate all if not ind.fitness.valid]
        
    ### logger.info('_________________________________')
    ### logger.info(str(len(invalid_ind))+' fitness calculations for initial generation...')
    if not skip_checkpoint_eval:
        evaluate_result = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, evaluate_result):
            ind.fitness.values = fit[0]
            ind.fitness.cvalues = fit[1]
            ind.fitness.n_constraints = len(ind.fitness.cvalues)
            # Allow for additional info to be saved (for example, a dictionary of properties)
            if len(fit) > 2:
                ind.fitness.info = fit[2]
    ### logger.info(str(len(invalid_ind))+' fitness calculations for initial generation...DONE')
        write_txt_population(population, os.path.join(output_dir, 'initial_pop.txt') ) 
    
      
        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        population = toolbox.select(population, len(population))
       
        record = stats.compile(population)
        logbook.record(gen=start_gen, evals=len(invalid_ind), **record)
        print(logbook.stream)   

    # Initial history update. Subsequent updates will be made when mating and mutating
    if do_history:
        history.update(population)

    # Begin the generational process
    for gen in range(start_gen, NGEN):

        if os.path.exists(abort_file):
            print('abort_file detected: ', abort_file)
            return population, logbook

        # Vary the population
        offspring = tools.selTournamentDCD(population, len(population))
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                #print(ind1)
                #print(ind2)
                toolbox.mate(ind1, ind2)
            
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
               
               
        ### logger.info(str(len(invalid_ind))+' fitness calculations for generation '+str(gen)+' ...')
        tstart = time.time()
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[0]
            ind.fitness.cvalues = fit[1]
            ind.fitness.n_constraints = len(ind.fitness.cvalues)
            # Allow for additional info to be saved (for example, a dictionary of properties)
            if len(fit) > 2:
              ind.fitness.info = fit[2]
        tend = time.time()
        ### logger.info(str(len(invalid_ind))+' fitness calculations for generation '+str(gen)+' DONE, time = '+str(tend-tstart)+' s')
        
        ### logger.info('Population selection for generation '+str(gen)+' ...')
        # Select the next generation population
        population = toolbox.select(population + offspring, MU)
        
        # Sort for convenience
        population.sort(key=lambda x: x.fitness.values)
        
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)
        ### logger.info('Population selection for generation '+str(gen)+' DONE')

        gen_name = 'pop_'+str(gen)
        write_txt_population(population, os.path.join(output_dir, gen_name+'.txt') ) 
        
        if do_history:
            pickle.dump(dict(history=history), open(os.path.join(output_dir, 'history.pkl'), 'wb'))
        
        #  CHECKPOINT
        if gen % CHECKPOINT_FREQUENCY == 0:
            filename = gen_name+'.pkl'
            write_checkpoint(os.path.join(output_dir, filename), population=population, generation=gen, logbook=logbook)
            ### logger.info('_________________________________')
            ### logger.info('Checkpoint '+filename+' written')
            
    return population, logbook
        

