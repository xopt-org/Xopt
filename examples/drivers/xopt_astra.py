#!/usr/bin/env python
# coding: utf-8


"""
xopt Astra driver

Basic usage:

mpirun -n 64 python -m mpi4py.futures xopt_astra.py xopt.in


"""

from xopt import nsga2_tools, legacy
from xopt.tools import xopt_logo, load_vocs, full_path, add_to_path
from xopt.evaluators.astra_evaluate import xopt_evaluate_astra_with_generator, xopt_archive_astra_h5
import xopt.configure

from pprint import pprint

import argparse
import os
import sys

USE_MPI = True
if USE_MPI:
    from mpi4py.futures import MPIPoolExecutor
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()
else:
    mpi_rank = 0
    mpi_size = 1


ARGS = [sys.argv[-1]]
#ARGS = 'xopt.in'.split()

parser = argparse.ArgumentParser(description='Configure xopt')
parser.add_argument('input_file', help='input_file')
args = parser.parse_args(ARGS)
infile = args.input_file

assert os.path.exists(infile)


# Load config
config = xopt.configure.load_config(infile)
# Completely configure
config = xopt.configure.configure(config)   


# needs VOCS, plus some gpt_distgen_eval options
evaluate_options = {}
evaluate_options.update(config['vocs'])
evaluate_options.update(config['astra_config'])
nsga2_params = config['nsga2_config']
nsga2_params['output_dir'] = config['xopt_config']['output_dir']



def create_toolbox(vocs):
    """
    Returns a toolbox from a vocs dict. 
    """
    toolbox_params = legacy.toolbox_params(variable_dict=vocs['variables'], constraint_dict = vocs['constraints'], objective_dict = vocs['objectives'])
    toolbox = nsga2_tools.nsga2_toolbox(**toolbox_params)
    return toolbox
toolbox = create_toolbox(config['vocs'])


# Register the evaluate function
toolbox.register('evaluate', xopt_evaluate_astra_with_generator, **evaluate_options)
toolbox.register('archive', xopt_archive_astra_h5)



if __name__ == "__main__" and USE_MPI:
    print(xopt_logo)
    print('_________________________________')
    print('Parallel execution with', mpi_size, 'workers')   
    print('Configuration:')  
    pprint(config)
    sys.stdout.flush() 
    
    with MPIPoolExecutor() as executor:
        toolbox.register('map', executor.map) 
        nsga2_tools.main(toolbox, **nsga2_params)
        
elif __name__ == "__main__":
    print(xopt_logo)
    print('Serial execution')
    pprint(config)
    toolbox.register('map', map)    
    pprint(config)
    pop, stats = nsga2_tools.main(toolbox, **nsga2_params)
    pop.sort(key=lambda x: x.fitness.values)   
    print('------------ final stats ------------')
    print(stats)        

