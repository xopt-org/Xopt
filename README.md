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



