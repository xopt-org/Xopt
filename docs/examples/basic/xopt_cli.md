# Running Xopt from the Command Line

The tool `xopt-run` allows users without advanced needs to run optimizations using only a YAML config file and an evaluation function available from python's path.
You may use `xopt-run` if your optimization satisfies the following requirements.

- You are using one of Xopt's built in `Generator` classes.
- You plan to run the evaluation function in serial (vectorized or non-vectorized), or use `ProcessPoolExecutor` or `ThreadPoolExecutor` for parallelism.
- The parameter `Xopt.max_evaluations` is sufficient to define your stopping condition.

## Usage

Once Xopt is installed from conda/pip, the tool will be available in your system path and can be called like this.

```
xopt-run [-h] [--executor {map,thread_pool,process_pool}] [--max_workers MAX_WORKERS]
         [--python_path PYTHON_PATH] [--override OVERRIDE] config
```

The available arguments are the following.

- `config`: (required) Path to the YAML config file
- `-h`: Display help text for the tool.
- `--executor`: Override the executor defined in the config file. Available options are "map" (run serially, ie using the function `map`), "thread_pool" (use `ThreadPoolExecutor`), and "process_pool" (use `ProcessPoolExecutor`).
- `--max_workers`: Override the number of workers in Xopt and launch this number of threads/processes if using a parallel executor.
- `--python_path`: Add this directory to your python path. By default, the current working directory. May include environment variables and the character `~` to represent the user's home directory. Note: you can pass these values into the tool without shell expansion by surrounding them by single quotes. May have more than one of this flag.
- `--override`: Override a value from the YAML config file. May have more than one of this flag. More details below.

## Overriding Settings in Config File
The flag `--override` may be used to override values in the configuration file.
These overrides are defined using a simple key value format where each hierarchical object in the config file is separated by the period character (".").
For example, to set the parameter `eta_m` of the mutation operator in `NSGA2Generator` to 20, the flag `--override generator.mutation_operator.eta_m=20` may be used.

Overrides are applied to the config file before it used to generate the `Xopt` object.
Data types for the value follow YAML conventions.

## Making the Evaluation Function Acessible to `xopt-run`

Xopt will call the evaluation function defined in the YAML file under `evaluator.function`.
The value of this parameter should be written in the format `<package>.<function name>` where `<package>` is the name of a package available to `xopt-run`.
The package containing your evaluation function can either be installed in your python environment or be a file/directory sitting within your python path (ie following the conventions of `importlib`)
Xopt will automatically import the package and then use the specified function for optimizations.

The argument `--python_path <path>` is used to add the directory containing your python file with the evaluation function to your python path.
By default, the current working directory is included in your python path.

## Example Usage
### Calling an Evaluation Function from a Python File
As an example, if you would like to use a custom evaluation function called `eval_fun` and have defined it in the file `my_optimization.py`, your YAML config file would contain the following entry.
```yaml
evaluator:
  # The evaluation function
  function: my_optimization.eval_fun

  # Additional keyword arguments that are passed to the evaluation function
  function_kwargs:
    arg1: 1.0
    arg2: "a path"
```
You would then call the config file like this.
```
xopt-run --python_path <path to directory containing my_optimization.py> config.yaml
```
You do not need to include the argument `--python_path` if your python file is in the same directory in which you are running `xopt-run`.

### Solving a Test Problem
A complete example of solving the multi-objective optimization problem "ZDT3" using the algorithm NSGA2 is linked below.

[Solving ZDT3 with NSGA2Generator](../ga/nsga2/yaml_interface/index.md)
