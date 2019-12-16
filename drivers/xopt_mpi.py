#!/usr/bin/env python
# coding: utf-8


"""
xopt MPI driver

Basic usage:

mpirun -n 4 python -m mpi4py.futures $XOPT_DIR/drivers/xopt_mpi.py xopt.yaml


"""

from xopt import Xopt, xopt_logo

from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

import argparse
import os
import sys
from pprint import pprint


ARGS = [sys.argv[-1]]
#ARGS = 'xopt.in'.split()

parser = argparse.ArgumentParser(description='Configure xopt')
parser.add_argument('input_file', help='input_file')
args = parser.parse_args(ARGS)
infile = args.input_file

assert os.path.exists(infile), f'Input file does not exist: {infile}'


X = Xopt(infile)


if __name__ == "__main__":
    print(xopt_logo)
    print('_________________________________')
    print('Parallel execution with', mpi_size, 'workers')   
#    print('Configuration:')  
#    pprint(config)
    sys.stdout.flush() 
    
    with MPIPoolExecutor() as executor:
        X.run(executor=executor)
