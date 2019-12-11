export HDF5_USE_FILE_LOCKING=FALSE

export XOPT_DIR=~/Code/GitHub/xopt
mpirun -n 4 python -m mpi4py.futures $XOPT_DIR/drivers/xopt_sampler.py xopt_astra.in
