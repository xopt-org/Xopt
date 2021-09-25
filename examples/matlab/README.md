This directory contains a Matlab interface to the Xopt module.

Requires Matlab and Matlab parallel computing toolbox (see Matlab documentation for your release to check for compative Python version).
Tested with R2021b and Python v.3.8 or v.3.9

Uses Matlab Python Engine to call user-provided Matlab function for objective function calls.

Contents:
* RUNME.sh - unix example command-line example of Xopt (Matlab engines started in separate process)
* RUNME.bat - Windows version of above
* RUNME.mlx - Matlab live script. Example of running Xopt optimizer from within Matlab environment, and comparison with builtin Matlab optimizer.
  - RUNME.html is output of live script
  
From within the Matlab environment:
>> help Xopt % instructions for using Xopt class
- requires Xopt.m and xopt_fun.m files in search path

>> RUNME % Runs example using fitnessfcn_xopt.m as example objective function (further documentation within live script)