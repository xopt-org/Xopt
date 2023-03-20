Pre-Configured Generators in Xopt
===============
A number of algorithms are implemented in Xopt using the ```Generator``` class for 
out-of-the-box usage. 
Below is a 
description of the different generators that are available in Xopt and their target 
use cases.

RandomGenerator
===============
Generates random points in the input space according to ```VOCS```.

Bayesian Generators
==
All of the generators here use Bayesian optimization (BO) type methods to solve single 
objective, multi objective and characterization problems. Bayesian generators 
incorperate unknown constrianing functions into optimization based on what is 
specified in ```VOCS```

- ```ExpectedImprovementGenerator```: implements Expected Improvement single 
  objective BO. Automatically balances trade-offs between exploration and 
  exploitation and is thus useful for general purpose optimization. 
- ```UpperConfidenceBoundGenerator```: implements Upper Confidence Bound single 
  objective BO. Requires a hyperparameter ```beta``` that explicitly sets the tradeoff 
  between exploration and exploitation. Default value of ```beta=2``` is a good 
  starting point. Increase $\beta$ to prioritize exploration and decrease ```beta``` to 
  prioritize exploitation.
- ```BayesianExplorationGenerator```: implements the Bayesian Exploration algorithm 
  for function characterization. This algorithm selects observation points that 
  maximize model uncertainty, thus picking points that maximize the information gain 
  about the target function at each iteration. If the target function is found to be 
  more sensative to one parameter this generator will adjust sampling frequency to 
  adapt. Note: specifying ```vocs.objective[1]```
  to ```MAXIMIZE``` or ```MINIMIZE``` does not change the behavior of this generator.
- ```MOBOGenerator```: implements Multi-Objective BO using the 
  Expected Hypervolume Improvement (EHVI) acquisition function. This is an ideal 
  general purpose multi-objective optimizer when objective evaluations cannot be 
  massively parallelized (below 5-10 parallel evaluations).
- ```MGGPO```: implements Multi-Generation Gaussian Process Optimization using the 
  Expected Hypervolume Improvement (EHVI) acquisition function. This is an ideal 
  general purpose multi-objective optimizer when objective evaluations can be 
  massively parallelized (above 5-10 parallel evaluations).

Evolutionary Generators
=====
- ```CNSGAGenerator```: implements Continuous Non-dominated Sorted Genetic Algorithm 
  which as a good general purpose evolutionary algorithm used for solving 
  multi-objective optimization problems where evaluating the objective is relatively 
  cheap and massively parallelizable (above 5-10 parallel evaluations).

Extremum Seeking Generators
===
- ```ExtremumSeekingGenerator```: implements the Extremum Seeking algorithm which is 
  ideal for solving optimization problems that are suceptable to drifts.

Scipy Generators
===
These generators serve as wrappers for algorithms implemented in scipy.
- ```NelderMeadGenerator```: implements Nelder-Mead (simplex) optimization.

Custom Generators
====
Any general algorithm can be implemented by subclassing the abstract ```Generator``` 
class and used in the Xopt framework. If you implement a generator for your use case 
please consider opening a pull request so that we can add it to Xopt! 