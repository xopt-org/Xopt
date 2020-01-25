# xopt
Simulation optimization, based on **DEAP** https://deap.readthedocs.io

Example run command, with $XOPT_DIR set to this cloned repository. 
```
mpirun -n 64 python -m mpi4py.futures $XOPT_DIR/drivers/xopt_mpi.py xopt.yaml
```

The complete configuration of a simulation optimization is given by a proper YAML file:

```yaml
xopt: {output_path: null, verbose: true, algorithm: cnsga}

cnsga: {max_generations: 50, population_size: 128, crossover_probability: 0.9, mutation_probability: 1.0,
  selection: auto, verbose: true, population: null}
  
simulation: 
  name: test_TNK
  evaluate: xopt.evaluators.test_TNK.evaluate_TNK  
  
vocs:
  name: TNK_test
  description: null
  simulation: test_TNK
  templates: null
  variables:
    x1: [0, 3.14159]
    x2: [0, 3.14159]
  objectives: {y1: MINIMIZE, y2: MINIMIZE}
  constraints:
    c1: [GREATER_THAN, 0]
    c2: [GREATER_THAN, 0]
  linked_variables: {x9: x1}
  constants: {a: dummy_constant}
```




Installing xopt
===============

Installing `xopt` from the `conda-forge` channel can be achieved by adding `conda-forge` to your channels with:

```shell
conda config --add channels conda-forge
```

Once the `conda-forge` channel has been enabled, `xopt` can be installed with:

```shell
conda install xopt
```

It is possible to list all of the versions of `xopt` available on your platform with:

```shell
conda search xopt --channel conda-forge
```



## Example simulations
xopt currently supports:

**ASTRA**

http://www.desy.de/~mpyflo/

**GPT**

http://pulsar.nl/gpt/index.html




## Cori (NERSC) setup

```
conda install -c conda-forge xopt
```
Follow instructions to build mpi4py:
https://docs.nersc.gov/programming/high-level-environments/python/
Note that there is a bug in Jupyterhub terminals. Type:
```
module swap PrgEnv-gnu PrgEnv-gnu
```
to get the C compiler activated. 

