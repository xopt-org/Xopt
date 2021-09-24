#!/bin/sh
# Script for running test_TNK example using Matlab calls for evaluation function, execute from examples/matlab directory

# Install python matlab_engine libs: https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

# Start 1 matlab_engine instance per core (use 4 cores) [this should run in a separate shell window]
xterm -e bash -ilc 'matlab -nodesktop -nosplash -r "parpool(4);matinit"' &
sleep 40

# Run example xopt instance with test_TNK_matlab.py evaluator which calls testeval.m function in running matlab_engines instances
# Here, we assume 4 cores, change to match the number of cores on your machine
mpirun -n 4 python -m mpi4py.futures -m xopt.mpi.run xopt_MOBO_matlab.yaml

