# xopt
Accelerator optimization, based on **DEAP** https://deap.readthedocs.io

Example run command, with $XOPT_DIR set to this cloned repository. 
```
mpirun -n 64 python -m mpi4py.futures $XOPT_DIR/drivers/xopt_gpt.py xopt_gpt.in
```

## Dependencies
```
python > 3.6
mpi4py > 3.0.0
h5py
numpy
```

## Supported Codes
xopt currently supports:

**ASTRA**

http://www.desy.de/~mpyflo/

**GPT**

http://pulsar.nl/gpt/index.html




## Cori (NERSC) setup

```
conda create -n xopt python=3 numpy scipy matplotlib h5py 
conda activate xopt
conda install -c conda-forge deap
```
Follow instructions to build mpi4py:
https://docs.nersc.gov/programming/high-level-environments/python/
Note that there is a bug in Jupyterhub terminals. Type:
```
module swap PrgEnv-gnu PrgEnv-gnu
```
to get the C compiler activated. 

