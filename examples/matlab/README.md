# Matlab interface to the Xopt module

* Requires Matlab and Matlab parallel computing toolbox (see Matlab documentation for your release to check for compative Python version).
* Tested with R2021b and Python v.3.8 or v.3.9
* Uses Matlab Python Engine to call user-provided Matlab function for objective function calls.

## Contents

* RUNME.mlx - Matlab live script. Example of running Xopt optimizer from within Matlab environment, and comparison with builtin Matlab optimizer.
* Xopt.m - Matlab class file to interface with Xopt (needs to be in Matlab search path)
* xopt_fun.m - Used by Xopt.m (needs to be in Matlab search path)
* TNK_constraints.m - Definition of constraints for TNK example used by Matlab optimizer gamultiobj
* TNK_opt.m - TNK example function and constraints used by Xopt
* test.py - Test file demonstrating use of Matlab python engine

## Run TNK optimization example from within Matlab

* internally generates xopt_eval.py & xopt_eval.yaml files in local directory at run time for running Xopt
```MATLAB
help Xopt % instructions for using Xopt class
RUNME % Runs example using fitnessfcn_xopt.m as example objective function (further documentation and results seen within live script)
```