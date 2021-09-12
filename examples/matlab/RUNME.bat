:: Script for running test_TNK example using Matlab calls for evaluation function, execute from root xopt directory

:: Install python matlab_engine libs: https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

:: Start 1 matlab_engine instance per core
matlab -nodesktop -nosplash -r "cd examples\matlab;matinit"

:: Run example xopt instance with test_TNK_matlab.py evaluator which calls testeval.m function in running matlab_engines instances
:: Here, we assume 10 cores, change to match the number of cores on your machine
mpiexec -n 10 python -m mpi4py.futures -m xopt.mpi.run examples\matlab\xopt_mat.yaml