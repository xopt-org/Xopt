<div align="center">
  <img src="docs/assets/Xopt-logo.png", width="200">
</div>




Xopt
====


**`Documentation`** |
------------------- |
[![Documentation](https://img.shields.io/badge/Xopt-documentation-blue.svg)](https://ChristopherMayes.github.io/Xopt/index.html)  |




Flexible optimization of arbitrary problems in Python.

The goal of this package is to provide advanced algorithmic support for arbitrary 
simulations/control systems with minimal required coding. Users can easily connect 
arbitrary evaluation functions to advanced algorithms with minimal coding with 
support for multi-threaded or MPI-enabled execution.

Currenty **Xopt** provides:

- optimization algorithms:
  - `cnsga` Continuous NSGA-II with constraints.
  - `bayesian_optimization` Single objective Bayesian optimization (w/ or w/o constraints, serial or parallel).
  - `mobo` Multi-objective Bayesian optimization (w/ or w/o constraints, serial or parallel).
  - `bayesian_exploration` Bayesian exploration.
  - `multi_fidelity` Multi-fidelity Single objective Bayesian optimization.
- sampling algorithms:
  - `random sampler`
- Convenient YAML/JSON based input format.
- Driver programs:
  - `xopt.mpi.run` Parallel MPI execution using this input format.

Xopt does **not** provide: 
- your custom simulation via an `evaluate` function.






Current release info
====================

| Name | Downloads | Version | Platforms |
| --- | --- | --- | --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-xopt-green.svg)](https://anaconda.org/conda-forge/xopt) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/xopt.svg)](https://anaconda.org/conda-forge/xopt) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/xopt.svg)](https://anaconda.org/conda-forge/xopt) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/xopt.svg)](https://anaconda.org/conda-forge/xopt) |



Configuring an Xopt run
===============
Xopt runs are specified via a dictionary that can be directly imported from a YAML file.

```yaml
xopt: 
  output_path: .

algorithm:
  name: cnsga
  options: 
    max_generations: 50 
    population_size: 128
    crossover_probability: 0.9
    mutation_probability: 1.0
    selection: auto
    population: null
  
simulation: 
  name: test_TNK
  evaluate: xopt.tests.evaluators.TNK.evaluate_TNK  
  
vocs:
  variables: 
    x1: [0, 3.14159]
    x2: [0, 3.14159]
  objectives:
    y1: MINIMIZE 
    y2: MINIMIZE
  constraints:
    c1: [GREATER_THAN, 0]
    c2: [LESS_THAN, 0.5]
  linked_variables:
    x9: x1
  constants:
    a: dummy_constant
```


Using MPI
===============
Example MPI run, with `xopt.yaml` as the only user-defined file:
```b
mpirun -n 64 python -m mpi4py.futures -m xopt.mpi.run xopt.yaml
```

The complete configuration of a simulation optimization is given by a proper YAML file:






Defining evaluation function
===============
Xopt can interface with arbitrary evaluate functions (defined in Python) with the 
following form:
```
evaluate(params[Dict]) -> Dict
```
Evaluate functions must accept a dictionary object that **at least** has the keys 
specified in `variables, constants, linked_variables` and returns a dictionary 
containing **at least** the 
keys contained in `objectives, constraints`. Extra dictionary keys are tracked and 
used in the evaluate function but are not modified by xopt.




Installing Xopt
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



Developers
==========


Clone this repository:
```shell
git clone https://github.com/ChristopherMayes/Xopt.git
```

Create an environment `xopt-dev` with all the dependencies:
```shell
conda env create -f environment.yml
```


Install as editable:
```shell
conda activate xopt-dev
pip install --no-dependencies -e .
```

