# YAML Interface

In this note, the use of `NSGA2Generator` through its YAML config file interface is demonstrated.
We will present an example of running the optimizer on ZDT3, a simple biobjective test problem.

## Companion Notebook and Files

Use of the optimizer will require us to write several files, run some terminal commands, and then read the data back into python.
All of this is demonstrated in the notebook that goes with this example which may be reached through the link below.
Please open it in a new tab now to follow along with the rest of this document.
All of the files used in the tutorial may also be downloaded in the below archive.

- [Companion Notebook](nsga2_yaml.ipynb)
- [Tutorial Files Archive](assets/yaml_runner_example.zip)

## Defining the Problem

First, we must write the python function which will compute the objectives and constraints for our test problem.
In Xopt convention, this function (called the evaluation function) will accept a python dictionary and return a python dictionary.
The names and bounds/directions of the decision variables, objectives, and constraints are defined in the Xopt `VOCS` object.
We will handle this in the section after this one.

Let's write out a function that will compute the test problem ZDT3.
Place this in a file `eval_fun.py` sitting in the project directory for our example.
Xopt must be able to import the file containing your function and so it must be in your current working directory, installed in your python environment as a package, or otherwise in your python path.

```python
"""
eval_fun.py - Define the evaluation function

This file will be imported by xopt and supply the optimizer with the evaluation function to call
"""

import numpy as np


def eval_fun(in_dict: dict, n: int = 30) -> dict:
    """
    The function is ZDT3 from [1]. It is implemented using numpy vectorized operations to allow its use in
    quick example problems.

    [1] Zitzler, E., Deb, K., & Thiele, L. (2000). Comparison of Multiobjective Evolutionary Algorithms: Empirical Results.
        Evolutionary Computation, 8(2), 173–195.

    Parameters
    ----------
    in_dict : dict
        The dictionary of decision variables x1, x2, ..., x30
    n : int
        Number of decision variables, by default 30

    Returns
    -------
    dict
        The dictionary of objectives (f1, f2)
    """
    # Unpack the decision var dict
    x = np.array([in_dict[f"x{idx}"] for idx in range(1, n+1)])

    # Calculate objectives
    g = 1 + 9 * np.sum(x[1:], axis=0) / (n - 1)
    ret = {
        "f1": x[0].tolist(),
        "f2": (g * (1 - np.sqrt(x[0] / g) - x[0] / g * np.sin(10 * np.pi * x[0]))).tolist(),
    }
    return ret
```

The function `eval_fun` accepts a dict of the decision variables `x1`, `x2`, `x3`, and so on up until `n` decision variables.
It will then return the objectives `f1` and `f2`.
This example does not have constraints.
Note that for this particular example, we have defined the function in such a way that each decision variable may be a list of values which is interpreted as multiple individuals which will be evaluated at the same time.
This is important for the performance of this simple example, but many common applications will accept scalar values for decision variables instead.

The function has now been defined, but this is only half of the work in setting up a problem.
We must also describe to the optimizer the names of all decision variables, objectives, and constraints and their bounds and directions.
This happens in the Xopt configuration file which is described in the next section.

## Xopt Configuration

Xopt problems may be set up through a YAML configuration file.
This file defines which optimization algorithm to use, what hyperparameters it has, the problem, and details about how the individuals will be evaluated.
The configuration file used to solve ZDT3 (n=30) with the `NSGA2Generator` is included below.
Write this in a file in your project directory called `nsga2_zdt3.yml`.

```yaml
max_evaluations: 5000

generator:
  name: nsga2  # Use `NSGA2Generator`
  population_size:  50  # Number of individuals in each population
  output_dir: nsga2_output  # Where data will be output, remove to not output anything

evaluator:
  # Xopt will import the module `eval_fun` and use the function `eval_fun` in it
  # Note: we must call the Xopt runner script within the directory containing `eval_fun.py` for this to work
  function: eval_fun.eval_fun

  # Keyword arguments which are passed to the evaluator function
  function_kwargs:
    n: 30

  # Send the function multiple decision vars at a time? How many individuals to evaluate at one time. Note: we set it
  # to the population size so that a full generation is evaluated at once.
  vectorized: true
  max_workers: 50

# Define the variables, objectives and constraints
vocs:
  # Each decision variable with its lower and upper bound
  variables:
    x1: [0, 1]
    x2: [0, 1]
    x3: [0, 1]
    x4: [0, 1]
    x5: [0, 1]
    x6: [0, 1]
    x7: [0, 1]
    x8: [0, 1]
    x9: [0, 1]
    x10: [0, 1]
    x11: [0, 1]
    x12: [0, 1]
    x13: [0, 1]
    x14: [0, 1]
    x15: [0, 1]
    x16: [0, 1]
    x17: [0, 1]
    x18: [0, 1]
    x19: [0, 1]
    x20: [0, 1]
    x21: [0, 1]
    x22: [0, 1]
    x23: [0, 1]
    x24: [0, 1]
    x25: [0, 1]
    x26: [0, 1]
    x27: [0, 1]
    x28: [0, 1]
    x29: [0, 1]
    x30: [0, 1]

  # Name of the objectives and which direction they are (MINIMIZE or MAXIMIZE)
  objectives:
    f1: MINIMIZE
    f2: MINIMIZE

  # We don't have constraints, but an example of what their definition looks like is included below
  # constraints:
  #   g1: ["LESS_THAN", 0]
  #   g2: ["GREATER_THAN", 0]

  # Constants may also be passed, these are used within the evaluation function
  # constants:
  #   const1: 0.1
  #   const2: "path/to/template/file.txt"
```

The topmost line defines the stopping condition.
The optimizer will terminate after completing 5000 evaluations of the function.
```yaml
max_evaluations: 5000
```

Next, we setup the generator.
The `NSGA2Generator` will be used.
This class implements the popular NSGAII multiobjective optimization algorithm.
Some configuration of the algorithm is included.
The population size is set to 50 and output is turned on at the specified path.

```yaml
generator:
  name: nsga2  # Use `NSGA2Generator`
  population_size:  50  # Number of individuals in each population
  output_dir: nsga2_output  # Where data will be output, remove to not output anything
```

The next section tells Xopt which function to call for evaluating individuals and how it should be called.
We point to `eval_fun.eval_fun` meaning the function `eval_fun` within the module `eval_fun`.
Xopt will automatically import the module `eval_fun` which it will find as long as the optimizer code is run from within the same directory as `eval_fun.py`.
We may pass keyword arguments to the function through `function_kwargs` which we use to set the number of decision variables.
The last two settings tell Xopt to pass multiple individuals to the evaluator at the same time.
Here, we evaluate one full generation at a time for performance reasons.

```yaml
evaluator:
  # Xopt will import the module `eval_fun` and use the function `eval_fun` in it
  # Note: we must call the Xopt runner script within the directory containing `eval_fun.py` for this to work
  function: eval_fun.eval_fun

  # Keyword arguments which are passed to the evaluator function
  function_kwargs:
    n: 30

  # Send the function multiple decision vars at a time? How many individuals to evaluate at one time. Note: we set it
  # to the population size so that a full generation is evaluated at once.
  vectorized: true
  max_workers: 50
```

Lastly, the variables, objectives, and constraints (VOCs) are defined.
Each variable is listed by the name the evaluation function expects to find in the input dictionary.
The lower and upper bound of each variable is also given.
The objectives are defined using the names the evaluator function will return them by as well as whether they are to be minimized or maximized.
Although they are not used here, we included a commented out section of the constraints.
One would again define them by the name the evaluation function will emit them with and then the direction of the constraint and its boundary are set.
Lastly, constants which are simply passed in the input dictionary are allowed.

```yaml
# Define the variables, objectives and constraints
vocs:
  # Each decision variable with its lower and upper bound
  variables:
    x1: [0, 1]
    x2: [0, 1]
    x3: [0, 1]
    ...

  # Name of the objectives and which direction they are (MINIMIZE or MAXIMIZE)
  objectives:
    f1: MINIMIZE
    f2: MINIMIZE

  # We don't have constraints, but an example of what their definition looks like is included below
  # constraints:
  #   g1: ["LESS_THAN", 0]
  #   g2: ["GREATER_THAN", 0]

  # Constants may also be passed, these are used within the evaluation function
  # constants:
  #   const1: 0.1
  #   const2: "path/to/template/file.txt"
```

## Running Xopt

We are now ready to run the optimization!
Your directory should include the following files.
- `eval_fun.py`
- `nsga2_zdt3.yml`

To run the optimization, execute the following command from the same directory as `eval_fun.py` using a python environment with Xopt installed.
```bash
xopt-run nsga2_zdt3.yml
```
The CLI tool `xopt-run` is a general tool to run Xopt from a YAML configuration file for users with basic parallelization needs.
Read more about it on the [Xopt CLI documentation page](../../../basic/xopt_cli.md).
It should complete in roughly 30s and produce a directory called `nsga2_output` in the current location.

Navigate to the output directory and observe the files there.
- `populations.csv`: Each completed population is recorded to this file
- `data.csv`: Contains all evaluated inviduals
- `log.txt`: A record of all log messages the genreator emitted during its run
- `vocs.txt`: A copy of the variable, objectives, and constraints (VOCs) definitions
- `checkpoints/`:  This directory contains checkpoint files which are used with the `checkpoint_file` key of the generator to restart an optimization.


## Analyzing Data

All data from this optimization run is saved to the output directory.
The file `data.csv` contains each evaluated individual and the file `populations.csv` has every generation (identified by the index `xopt_generation`).
These files may be easily loaded and processed inside of python using the library `pandas`.
For example, the final generation can be be retrieved with this code.

```python
import pandas as pd

df = pd.read_csv("nsga2_output/populations.csv")
last_gen = df[df["xopt_generation"] == df["xopt_generation"].max()]
```

To see the results of the plotting the final generation, please refer to the companion notebook linked above.
A link is also included here for convenience.

- [Companion Notebook](nsga2_yaml.ipynb)

## Restarting the Optimization from a Checkpoint

When configured to use file output, `NSGA2Generator` will periodically save a full copy of its state in a checkpoint file.
How often this happens can be controlled by `checkpoint_freq`, with every generation being the default.
The checkpoint files allow the generator to be reloaded with all of the same information it had at the moment it was saved.
The optimization can then be continued from this point on with no loss of information.

Use the key `checkpoint_file` in the YAML file under the `generator` section and set its value to the path of the checkpoint you would like to use to load it.
Extra keys in the generator section (such as `population_size`) will override the corresponding setting in the checkpoint file allowing you to tweak parameters as the optimization continues.
For more detail, please refer to the section on running the generator from a checkpoint file in the companion notebook.

```yaml
generator:
  name: nsga2  # Use `NSGA2Generator`
  checkpoint_file: nsga2_output/checkpoints/20250805_065102_1.txt  # Path to the checkoint to start from
  output_dir: nsga2_from_checkpoint_output  # Where data will be output, this overrides checkpoint settings
```
