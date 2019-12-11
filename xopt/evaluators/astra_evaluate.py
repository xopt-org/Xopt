
from xopt import legacy
from xopt.tools import decode1, write_attrs


from astra import run_astra_with_generator, writers
from astra.astra_calc import calc_ho_energy_spread

import numpy as np
import time
import sys




def end_output_data(output):
    """
    Some outputs are lists. Get the last item. 
    """
    o = {}
    for k in output:
        val = output[k]
        if isinstance(val, str): # Encode strings
            o[k] = val.encode()
        elif np.isscalar(val):
            o[k]=val
        else:
            o['end_'+k]=val[-1]
           
    return o



def default_astra_merit(A):
    """
    merit function to operate on an evaluated LUME-Astra object A. 
    
    Returns dict of scalar values
    """
    # Check for error
    if A.error:
        return {'error':True}
    else:
        m= {'error':False}
    
    # Gather output
    m.update(end_output_data(A.output))
    
    # Load final screen for calc
    A.load_screens(end_only=True)
    screen = A.screen[-1]        
    # TODO: time in screen isn't correct??? 
    m['end_higher_order_energy_spread'] = calc_ho_energy_spread( {'t':screen['z_rel'], 'Energy':(screen['pz_rel'])*1e-3},verbose=False) # eV
    
    # Lost particles have status < -6
    nlost = len(np.where(screen['status'] < -6)[0])    
    m['end_n_particle_loss'] = nlost
          
    return m


def astra_data_for_archiving(A):
    """
    Return dict of archive items
    """
    d = {}
    d['input'] = A.input
    d['output'] = A.output
    #d['log'] = A.log
    d['end_output'] = end_output_data(A.output)
    if len(A.screen) >0:
        d['final_screen'] = A.screen[-1]
    return d


def evaluate_astra_with_generator(settings, **options):
    """
    
    """

    # Pop id
    if 'run_id' in settings:
        run_id = settings.pop('run_id')
    else:
        # Hash settings for run_id
        run_id = abs(hash(str(settings)))
        #run_id = 999
    
    # Defaults
    run_options = {
        'merit_fun': default_astra_merit,
        'workdir':None,
        'verbose':False,
        'timeout':None,
        'astra_bin':'$ASTRA_BIN',
        'generator_bin':'$GENERATOR_BIN'
    }
    # Pick up all options
    run_options.update(options)

    # This will contain all output data, for archiving. 
    output = {'settings':settings,
              'templates':run_options['templates']}
    
    # Run
    t1 = time.time()
    try:
        A = run_astra_with_generator(settings=settings, 
                                     astra_bin=run_options['astra_bin'],
                                     generator_bin=run_options['generator_bin'],
                                     astra_input_file=run_options['templates']['astra'],
                                     generator_input_file=run_options['templates']['generator'],
                                     workdir=run_options['workdir'],
                                     verbose=run_options['verbose'],
                                     timeout=run_options['timeout']
                                    )
        # Merit calc
        merits = run_options['merit_fun'](A)

        # Add archive data
        output.update(astra_data_for_archiving(A))
        
        error = A.error
        
    except Exception as ex:
        print('Exception in evaluate_astra_with_generator:', ex)
        print('Settings:', settings)
        error = True
        merits = {'error':True}
    finally:
        total_time = time.time() - t1
               
    if not error:
        print('Good Astra run', total_time)

    
    # Add merits
    output['merits'] = merits 
    
    # Add run info
    output['run'] = {'start_time':t1,
                    'run_time': total_time,
                    'error': error,
                  #  'mpi_rank':mpi_rank,
                    'run_id': run_id
                    }
     
    # Flush print
    sys.stdout.flush()
    
    return output
    
    
def xopt_evaluate_astra_with_generator(individual, **options):
    """
    
    """
    # Decode vector
    vec = np.array([float(x) for x in individual] )
    settings = decode1(vec,legacy.sorted_names(options['variables']))
    # Add constants
    settings.update(options['constants'])
    # Add linked variables
    if 'linked_variables' in options:
        for k, v in options['linked_variables'].items():
            settings[k] = settings[v]
    
    # Actual eval
    output = evaluate_astra_with_generator(settings, **options)
    
    # These are used in objectives, constraints evaluation
    merits = output['merits']
    
    # If error, return    
    error = output['run']['error']
    if error:
        print('Astra run error!')
        objectives =  [0.0 for o in options['objectives']]
        constraints = [-666.0 for x in options['constraints']]
        return (objectives, constraints, output)
    
    objectives = legacy.evaluate_objectives(options['objectives'], merits)
    constraints = legacy.evaluate_constraints(options['constraints'], merits)    
    
    return (objectives, constraints, output)


#----------------------------------------
# Archiving

def archive_astra_h5(h5, output, name=None):  
    """
    Archives Astra output data to h5
    """
    if name:
        g = h5.create_group(name)
    else:
        g = h5
    
    # inputs
    for key in ['run', 'settings', 'templates']:
        if key in output:
            write_attrs(g, key, output[key])
    
    if output['run']['error']:
        return
    
    # Simple data
    for key in ['merits', 'end_output']:
        if key in output:
            write_attrs(g, key, output[key])  
            
    # Astra input file data
    writers.write_input_h5(g, output['input'])
   
    # Astra output
    writers.write_output_h5(g, output['output'])
    
    # Only write screen if it exists
    if 'final_screen' in output:
        writers.write_astra_particles_h5(g, 'final_screen', output['final_screen'])  
    
    
    
    
def xopt_archive_astra_h5(h5, individual):
    """
    Write all fitness.info from an individual 
    """
    run_id = individual.id
    name = 'ind_'+str(run_id)
    
    if not 'fitness' in dir(individual):
        print('Individual has no fitness. id:', run_id)
        return
    
    if not 'info' in dir(individual.fitness):
        print('Individual has nothing to archive. id:', run_id)
        return
    
    output = individual.fitness.info
    
    # Overwrite this
    output['run']['run_id'] = run_id
    
    archive_astra_h5(h5, output, name = name)
                  