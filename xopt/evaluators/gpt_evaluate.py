from pisapy.eval_parser import eval_parser
from pisapy.gpt_distgen_eval import gpt_distgen_eval
from pisapy.gpt import calc_statistical_data
from pisapy.writers import  write_gpt_particles_h5
import numpy as np
import platform
import h5py
import time
import json
import os
from tempfile import TemporaryDirectory
from shutil import rmtree

from xopt.tools import decode1
from xopt import legacy


#******************************************************************************************
# This is the user defined MERIT function (Objectives, Constraints), edit here 
#******************************************************************************************
def cbeta_merit_fun(variables, data):
    """
    CBETA merit function for use in gpt_distgen_eval. 
    """
    merits = {}
    merits['error']=False
    merits['scalars'] = {} 
    merits['screens'] = {}

    NSTART = variables["particle_count"]  # Number of particles at the start of tracking

    try:

        # We the need merit function computation here...for now, will just do the minimum work
        # Analyze last screen phase space output
   
        screens = {}  
        # Special named screens
        
        HSLIT_z = 12.35187788493322   # Position
        VSLIT_z = 12.41949269277345
        A1_SCREEN_z = 0.922
        
        for screen in data.pdata:
            z = screen['z'].mean()
            if(np.abs(z - HSLIT_z) < 1e-6 ):
                screens['dl_horizontal_slit'] = screen
            if(np.abs(z - VSLIT_z) < 1e-6 ):
                screens['dl_vertical_slit'] = screen
            if(np.abs(z - A1_SCREEN_z) < 1e-6 ):
                screens['a1_screen'] = screen                
        
        # Find final screen
        zmax = -1
        n_screens = 0
        for key in screens:
            z = screens[key]['position']
            n_screens += 1
            if z > zmax:
                zmax = z
                final_screen = screens[key]

        if(n_screens == 0):
            raise ValueError('No screens found')                
                
        NSTART = variables["particle_count"]
        NLOST = NSTART - len(final_screen['t'])                
        
        if 'dl_vertical_slit' in screens:
            # Special calc for diagnostic line screens
            screenHslit = screens['dl_horizontal_slit']
            screenVslit = screens['dl_vertical_slit']                      
            enx = screenHslit['twiss']['x']['en']      # m-rad
            eny = screenVslit['twiss']['y']['en']  # m-rad
            stdtx = screenHslit['t'].std()         
            stdty = screenVslit['t'].std()
            stdx = screenHslit['x'].std()
            stdy = screenVslit['y'].std()         
        else:
            # Normal calc
            enx = final_screen['twiss']['x']['en']   # m-rad
            eny = final_screen['twiss']['x']['en']   # m-rad
            stdtx = final_screen['t'].std()         # s
            stdty = stdtx
            stdx = final_screen['x'].std()
            stdy = final_screen['y'].std()
        stdxy = 0.5*(stdx + stdy)      
        qb = np.abs(variables["total_charge"])                                        
        
       # Compute statistical data as a function of t,s
        tstat,pstat = calc_statistical_data(data,"./", use_ref=True, s_interpolation=True)
        merits['tstat']=tstat
        merits['pstat']=pstat
       
       # Scalars kept together
        merits['scalars']['n_particles_lost'] = NLOST
        merits['scalars']['norm_emit_x'] = enx
        merits['scalars']['norm_emit_y'] = eny
        merits['scalars']['max_norm_emit'] = max(enx, eny)
        merits['scalars']['stdt_x'] = stdtx
        merits['scalars']['stdt_y'] = stdty
        merits['scalars']['max_stdt'] = max(stdtx, stdty)
        merits['scalars']['std_x'] = stdx
        merits['scalars']['std_y'] = stdy        
        merits['scalars']['std_xy'] = stdxy    
        merits['scalars']['total_charge'] = qb*1e-12
         
       # Final screens  
        for s in screens:
            merits['screens'][s] = screens[s] 
        
    except Exception as ex:
        print('ERROR occured in user_merit_fun:',str(ex))
        merits['error']=True
        
    return merits




def xopt_gpt_distgen_eval(settings, **options):  
    """
    xopt wrapper for pisapy's gpt_distgen_eval, with the same options. 
    """  

    # Pop id
    if 'run_id' in settings:
        run_id = settings.pop('run_id')
    else:
        run_id = 999
    
    startdir = os.getcwd()
    # Defaults
    run_options = {
              'merit_fun': cbeta_merit_fun}
    # Pick up all options
    run_options.update(options)
    
    t1 = time.time()
    # Run in temporary directory
    run_options['workdir'] = TemporaryDirectory(prefix='gpt_', dir=options['workdir']).name

    try:
        all_output = gpt_distgen_eval(settings,  **run_options)
        error = False
    except:
        'exception: gpt_distgen_eval'
        all_output = {}
        error = True
    finally:
        total_time = time.time() - t1
         # Restore starting dir    
        os.chdir(startdir)
        # Make sure this is really cleaned up
        if os.path.exists(run_options['workdir']):
            print('Extra workdir cleanup:', run_options['workdir'])
            rmtree(run_options['workdir'])

    # Pick items from all_output
    output = {}
    
    # Add input settings
    output['settings'] = settings
    
    # check for errors
    
    if 'merits' not in all_output:
        output['merits'] = {}
        output['merits']['scalars'] = {'error':True}
        error = True
    elif all_output['merits']['error'] or all_output['error']:
        output['merits'] = {}
        output['merits']['scalars'] = {'error':True}        
        error = True
    else:
        output['merits'] = all_output['merits']
    
    
    if 'exception' in all_output:
        print('GPT exception encountered: ', all_output['exception'])
    
    # Add Run info
    run = output['run'] = {}
    run['error']= error
    run['run_id'] = run_id
    run['start_time'] = t1
    run['run_time'] = total_time
    #run['mpi_rank'] = mpi_rank
    
    #print('calc1 for ', mpi_rank, 'finished')
    
    return output
    
def evaluate_gpt(individual, **options):
    """
    xopt evaluate for a GPT run. 
    """
    
    # Extract the vector fron the individual
    vec = np.array([float(x) for x in individual] )
    
    settings = decode1(vec,legacy.sorted_names(options['variables']))
    
    # Add constants
    settings.update(options['constants'])
    
    output = xopt_gpt_distgen_eval(settings, template_dir=options['template_dir'], verbose=options['verbose'], workdir=options['workdir'], timeout=options['timeout']) 
    #print(output)
    scalars = output['merits']['scalars']
    run = output['run']
    error = run['error'] 
    
    if error:
        print('GPT run error!', run['run_time'])
        objectives =  [0.0 for o in options['objectives']]
        constraints = [-666.0 for x in options['constraints']]
    else:
        print('GPT run time', run['run_time'])
        objectives = legacy.evaluate_objectives(options['objectives'], scalars)
        constraints = legacy.evaluate_constraints(options['constraints'], scalars)
        
    # add settings
    output['settings'] = {}
    output['settings'].update(settings)        
    
    # Flush print
    sys.stdout.flush()
    
    # Add rank
    ## output['mpi_rank'] = mpi_rank
    
    # Add vec for debug
    #output['vec_debug'] = vec
    
    return (objectives, constraints, output)


    
#------ Archiving

def write_attrs(h5, group_name, data):
    """
    Simple function to write dict data to attribues in a group with name
    """
    g = h5.create_group(group_name)
    for key in data:
        g.attrs[key] = data[key]
    return g
    
    
def archive_gpt(h5, individual):
    """
    Write all fitness.info from an individual 
    """
    
    run_id = individual.id
    name = 'ind_'+str(run_id)
    g = h5.create_group(name)
    
    output = individual.fitness.info
    
    # input
    write_attrs(g, 'settings',  output['settings'])
    
    # Run info
    run = output['run']
    run['run_id'] = run_id # Add id
    write_attrs(g, 'run', run)
    
    if output['run']['error']:
        return
    
    # Merits (custom data)
    merits = output['merits'] 

    # Scalars
    write_attrs(g, 'scalars',  merits['scalars'])
 
    # Datasets 
    if 'tstat' in merits:
        tstat = merits['tstat']
        g2 = g.create_group('tstat')
        for key in tstat:
            g2[key] = tstat[key]
            
    # Screens  
    if 'screens' in merits:
        g2 = g.create_group('screens')
        screens = merits['screens']
        for key in screens:
             write_gpt_particles_h5(g2, key, screens[key])  
            
    # Clean info
    ## del individual.fitness.info    
    
    
    




