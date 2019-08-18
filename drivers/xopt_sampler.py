#!/usr/bin/env python
# coding: utf-8

# # xopt Astra sampler

# In[1]:


"""
xopt sampler

Basic usage:

mpirun -n 64 python -m mpi4py.futures xopt_sampler.py xopt.in

"""


# In[2]:


from xopt.tools import xopt_logo, load_vocs, random_settings, full_path, add_to_path, write_attrs_nested
from xopt.evaluators.astra_evaluate import evaluate_astra_with_generator, archive_astra_h5
import xopt.configure

import numpy as np
import h5py

from pprint import pprint

import argparse
import os
import sys
import time


# In[3]:


DEBUG = False

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
# Random seeds for each rank
np.random.seed(np.random.randint(np.iinfo(np.uint32).min,np.iinfo(np.uint32).max)+mpi_rank)


# In[5]:


ARGS = [sys.argv[-1]]
# Use in notebook:
#ARGS =  'xopt_astra.in'.split()

parser = argparse.ArgumentParser(description='Configure xopt')
parser.add_argument('input_file', help='input_file')
args = parser.parse_args(ARGS)
infile = args.input_file

assert os.path.exists(infile)


# In[6]:


# Load config
config = xopt.configure.load_config(infile)
# Completely configure
config = xopt.configure.configure(config)   

N_SAMPLES = config['sampler_config']['max_samples']
CHUNK_SIZE = config['sampler_config']['chunk_size']


# In[7]:


config


# In[8]:


# needs VOCS, plus some Astra eval options
evaluate_options = {}
evaluate_options.update(config['vocs'])
evaluate_options.update(config['astra_config'])


# In[9]:


def calc1(i):
    settings0 = random_settings(evaluate_options)
    if DEBUG:
        settings0['lspch'] = False
    output = evaluate_astra_with_generator(settings0, **evaluate_options)
    return output


# In[10]:


ROOT = config['xopt_config']['output_dir']

def get_h5_name(i, prefix='astra_samples_'):
    name = prefix+str(i)+'.h5'
    return os.path.join(ROOT, name)
#get_h5_name(0)


# In[11]:



counter = 0
def new_filename():
    global counter
    while True:
        name = get_h5_name(counter)
        if os.path.exists(os.path.join(ROOT, name)):
            counter  += 1
        else:
            break
    return name
#new_filename()


# In[ ]:


if __name__ == "__main__" and USE_MPI:
    print(xopt_logo)
    print('_________________________________')
    print('Sampler')
    print('.......')
    print('Parallel execution with', mpi_size, 'workers')   
    print('Configuration:')  
    pprint(config)
    sys.stdout.flush() 
    
    with MPIPoolExecutor() as executor:
        temp_archive_name = os.path.join(ROOT, 'temp_samples.h5')
        
        h5 = h5py.File(temp_archive_name, 'w')
        
        t0 = time.time()
        t1 = t0
        # Map
        output = executor.map(calc1, range(N_SAMPLES), unordered=True )
        
        ii = 0
        for o1 in output:
            ii += 1
            #print(ii)
            #print('.', end='')
            name = hex(o1['run']['run_id'])
            
            archive_astra_h5(h5, o1, name)   
            
            if ii % CHUNK_SIZE == 0:
                h5.close()
                fname = new_filename()
                os.rename(temp_archive_name, fname) 
                
                t2 = time.time() 
                dt = t2 - t1
                t1 = t2
                
                print('total samples:', ii)
                print('archiving',CHUNK_SIZE, 'samples in', fname, 'time', dt)
                sys.stdout.flush()
                h5 = h5py.File(temp_archive_name, 'w')
                
        # Cleanup
        h5.close()
        os.rename(temp_archive_name, new_filename()) 
        
                    
elif __name__ == "__main__" :
    with open('sampler_log', 'w') as f:    
        temp_archive_name = os.path.join(ROOT, 'temp_samples.h5')
        
        h5 = h5py.File(temp_archive_name, 'w')
        
        # Map
        output =map(calc1, range(N_SAMPLES))
        
        ii = 0
        for o1 in output:
            ii += 1
            #print(ii)
            #print('.', end='')
            name = hex(o1['run']['run_id'])
            
            archive_astra_h5(h5, o1, name)   
            
            if ii % CHUNK_SIZE == 0:
                h5.close()
                fname = new_filename()
                os.rename(temp_archive_name, fname) 
                print('total samples:', ii)
                print('archiving',CHUNK_SIZE, 'samples in', fname)
                sys.stdout.flush()
                h5 = h5py.File(temp_archive_name, 'w')
                
        # Cleanup
        h5.close()
        os.rename(temp_archive_name, new_filename()) 

                        
        


# In[ ]:




