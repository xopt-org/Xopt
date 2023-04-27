<div align="center">
  <img src="assets/Xopt-logo.png", width="200">
</div>




Xopt
===============

Flexible optimization of arbitrary problems in Python.

The goal of this package is to provide advanced algorithmic support for arbitrary 
simulations/control systems with minimal required coding. Users can easily connect 
arbitrary evaluation functions to advanced algorithms with minimal coding with 
support for multi-threaded or MPI-enabled execution.

Currenty **Xopt** provides:

- optimization algorithms:
     - `cnsga` Continuous NSGA-II with constraints.
     - `upper_confidence_bound` Single objective Bayesian optimization (w/ or w/o 
       constraints, serial or parallel).
     - `mobo` Multi-objective Bayesian optimization (w/ or w/o constraints, serial or parallel).
     - `bayesian_exploration` Bayesian exploration.
- sampling algorithms:
     - `random sampler`
- Convenient YAML/JSON based input format.
- Driver programs:
     - `xopt.mpi.run` Parallel MPI execution using this input format.

 **Xopt** does **not** provide: 
- your custom simulation via an `evaluate` function.

Rather, **Xopt** asks you to define this function.

Getting Started
===============
[Xopt Overview PDF](assets/xopt_overview.pdf) gives an overview of Xopt's design and 
usage.

[Xopt Built-In Generators](algorithms.md) provides a list of available algorithms 
implemented in the Xopt ```Generator``` framework.

[Simple Bayesian Optimization Example](examples/bayes_opt/bo_tutorial.ipynb) shows 
Xopt usage for a simple optimization problem.


Configuring an Xopt run
===============
Xopt runs are specified via a dictionary that can be directly imported from a YAML 
file.

```yaml
xopt:
    max_evaluations: 6400

generator:
    name: cnsga
    population_size: 64
    population_file: test.csv
    output_path: .

evaluator:
    function: xopt.resources.test_functions.tnk.evaluate_TNK
    function_kwargs:
      raise_probability: 0.1

vocs:
    variables:
        x1: [0, 3.14159]
        x2: [0, 3.14159]
    objectives: {y1: MINIMIZE, y2: MINIMIZE}
    constraints:
        c1: [GREATER_THAN, 0]
        c2: [LESS_THAN, 0.5]
    constants: {a: dummy_constant}
```


Defining evaluation function
===============
Xopt can interface with arbitrary evaluate functions (defined in Python) with the 
following form:
```python
evaluate(inputs: dict) -> dict
```
Evaluate functions must accept a dictionary object that **at least** has the keys 
specified in `variables, constants, linked_variables` and returns a dictionary 
containing **at least** the 
keys contained in `objectives, constraints`. Extra dictionary keys are tracked and 
used in the evaluate function but are not modified by xopt.

Using MPI
===============
Example MPI run, with `xopt.yaml` as the only user-defined file:
```b
mpirun -n 64 python -m mpi4py.futures -m xopt.mpi.run xopt.yaml
```
