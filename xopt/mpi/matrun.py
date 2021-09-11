#!/usr/bin/env python
# coding: utf-8


"""
xopt MPI driver with Matlab calls for evaluator

Basic usage:

mpirun -n 4 python -m mpi4py.futures -m xopt.mpi.matrun xopt.yaml


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
import matlab.engine


ARGS = [sys.argv[-1]]
#ARGS = 'xopt.in'.split()

parser = argparse.ArgumentParser(description='Configure xopt')
parser.add_argument('input_file', help='input_file')
args = parser.parse_args(ARGS)
infile = args.input_file

assert os.path.exists(infile), f'Input file does not exist: {infile}'


if __name__ == "__main__":
    print(xopt_logo)
    print('_________________________________')
    print('Parallel execution with', mpi_size, 'workers')   
    
    # Start a matlab engine for each worker
    eng1 = matlab.engine.start_matlab()
    eng2 = matlab.engine.start_matlab()
    eng=[eng1,eng2]
    
    X = Xopt(infile)
    print(X)
    sys.stdout.flush() 
    with MPIPoolExecutor() as executor:
        X.run(executor=executor,eng)