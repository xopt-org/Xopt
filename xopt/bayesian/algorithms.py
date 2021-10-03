from typing import Dict
from xopt.tools import get_function
from . import generators
from .asynch_optimize import asynch_optimize
from .optimize import optimize
from .models.models import create_multi_fidelity_model


def bayesian_optimize(vocs, evaluate_f,
                      n_steps=1,
                      n_initial_samples=1,
                      output_path=None,
                      custom_model=None,
                      executor=None,
                      restart_file=None,
                      initial_x=None,
                      verbose=True,
                      generator_options=None):
    """
       Multi-objective Bayesian optimization

       Parameters
       ----------
       vocs : dict
           Varabiles, objectives, constraints and statics dictionary,
           see xopt documentation for detials

       evaluate_f : callable
           Returns dict of outputs after problem has been evaluated

       n_steps : int, default = 1
           Number of optimization steps to execute

       n_initial_samples : int, defualt = 1
           Number of initial samples to take before using the model,
           overwritten by initial_x

       output_path : str, default = ''
           Path location to place outputs

       custom_model : callable, optional
           Function of the form f(train_inputs, train_outputs) that
           returns a trained custom model

       executor : Executor, optional
           Executor object to run evaluate_f

       restart_file : str, optional
           File location of JSON file that has previous data

       initial_x : list, optional
           Nested list to provide initial candiates to evaluate, overwrites n_initial_samples

       verbose : bool, defualt = True
           Print out messages during optimization

       generator_options : dict
           Dictionary of options for MOBO

       Returns
       -------
       results : dict
           Dictionary with output data at the end of optimization

       """

    try:
        # Required
        acq_func = get_function(generator_options.pop('acquisition_function', None))
    except ValueError:
        raise ValueError('acquisition_function is a required parameter of generator_options.')

    generator = generators.generator.BayesianGenerator(vocs,
                                                       acq_func,
                                                       **generator_options)
    return optimize(vocs,
                    evaluate_f,
                    generator,
                    n_steps,
                    n_initial_samples,
                    output_path,
                    custom_model,
                    executor,
                    restart_file,
                    initial_x,
                    verbose,
                    generator.tkwargs
                    )


def mobo(vocs, evaluate_f,
         ref=None,
         n_steps=1,
         n_initial_samples=1,
         output_path=None,
         custom_model=None,
         executor=None,
         restart_file=None,
         initial_x=None,
         verbose=True,
         generator_options=None):
    """
    Multi-objective Bayesian optimization

    Parameters
    ----------
    vocs : dict
        Varabiles, objectives, constraints and statics dictionary,
        see xopt documentation for detials

    evaluate_f : callable
        Returns dict of outputs after problem has been evaluated

    ref : list
        Reference point for multi-objective optimization.

    n_steps : int, default = 1
        Number of optimization steps to execute

    n_initial_samples : int, defualt = 1
        Number of initial samples to take before using the model,
        overwritten by initial_x

    output_path : str, default = ''
        Path location to place outputs

    custom_model : callable, optional
        Function of the form f(train_inputs, train_outputs) that returns
        a trained custom model

    executor : Executor, optional
        Executor object to run evaluate_f

    restart_file : str, optional
        File location of JSON file that has previous data

    initial_x : list, optional
        Nested list to provide initial candiates to evaluate,
        overwrites n_initial_samples

    verbose : bool, default = True
        Print out messages during optimization

    generator_options : dict
        Dictionary of options for MOBO

    Returns
    -------
    results : dict
        Dictionary with output data at the end of optimization

    """

    # Handle None
    generator_options = generator_options or {}
                 
    generator = generators.mobo.MOBOGenerator(vocs, ref, **generator_options)
    return optimize(vocs,
                    evaluate_f,
                    generator,
                    n_steps,
                    n_initial_samples,
                    output_path,
                    custom_model,
                    executor,
                    restart_file,
                    initial_x,
                    verbose,
                    generator.tkwargs
                    )


def bayesian_exploration(vocs, evaluate_f,
                         n_steps=1,
                         n_initial_samples=1,
                         output_path=None,
                         custom_model=None,
                         executor=None,
                         restart_file=None,
                         initial_x=None,
                         verbose=True,
                         generator_options=None):
    """
        Bayesian Exploration

        Parameters
        ----------
        vocs : dict
            Varabiles, objectives, constraints and statics dictionary, see xopt documentation for detials

        evaluate_f : callable
            Returns dict of outputs after problem has been evaluated

        n_steps : int, default = 1
            Number of optimization steps to execute

        n_initial_samples : int, defualt = 1
            Number of initial samples to take before using the model, overwritten by initial_x

        output_path : str, default = ''
            Path location to place outputs

        custom_model : callable, optional
            Function of the form f(train_inputs, train_outputs) that returns a trained custom model

        executor : Executor, optional
            Executor object to run evaluate_f

        restart_file : str, optional
            File location of JSON file that has previous data

        initial_x : list, optional
            Nested list to provide initial candiates to evaluate, overwrites n_initial_samples

        verbose : bool, defualt = True
            Print out messages during optimization

        use_gpu : bool, default = False
            Specify if GPU should be used if available

        generator_options : dict
            Dictionary of options for MOBO

        Returns
        -------
        results : dict
            Dictionary with output data at the end of optimization

        """
    
    # Handle None
    generator_options = generator_options or {}

    generator = generators.exploration.BayesianExplorationGenerator(vocs,
                                                                    **generator_options)
    return optimize(vocs,
                    evaluate_f,
                    generator,
                    n_steps,
                    n_initial_samples,
                    output_path,
                    custom_model,
                    executor,
                    restart_file,
                    initial_x,
                    verbose,
                    generator.tkwargs
                    )


def multi_fidelity_optimize(vocs, evaluate_f,
                            budget=1,
                            processes=1,
                            base_cost=1.0,
                            output_path=None,
                            custom_model=create_multi_fidelity_model,
                            executor=None,
                            restart_file=None,
                            initial_x=None,
                            verbose=True,
                            generator_options=None):
    """
        Multi-fidelity optimization using Bayesian optimization

        This optimization algorithm attempts to reduce the computational cost of
        optimizing a scalar function through the use of many low-cost approximate
        evaluations. We create a GP model that models the function f(x,c) where x
        represents free parameters and c in the range [0,1] reperestents the `cost`.
        This algorithm uses the knoweldge gradient approach
        (https://botorch.org/tutorials/multi_fidelity_bo) to choose points that are
        likely to provide the most information about the optimimum point at maximum
        fidelity (c=1).

        Since evaluations have variable cost, we specify a maximum `budget` that
        stops the algorithm when the total cost exceeds the budget. Cost for a single
        evaluation is given by `cost` + 'base_cost`.

        This algoritm is most efficient when used with a parallel executor,
        as evaluations will be done asynchronously, allowing multiple cheap
        simulations to run in parallel with expensive ones.

        Parameters
        ----------
        vocs : dict
            Varabiles, objectives, constraints and statics dictionary, see xopt
            documentation for detials

        evaluate_f : callable
            Returns dict of outputs after problem has been evaluated

        budget : int, default = 1
            Optimization budget

        processes : int, defualt = 1
            Number of parallel processes to use or number of candidates to generate
            at each step.

        base_cost : float, defualt = 1.0
            Base cost of running simulations. Total cost is base + `cost` variable

        output_path : str, default = ''
            Path location to place outputs

        custom_model : callable, optional
            Function of the form f(train_inputs, train_outputs) that returns a trained
            custom model

        executor : Executor, optional
            Executor object to run evaluate_f

        restart_file : str, optional
            File location of JSON file that has previous data

        initial_x : list, optional
            Nested list to provide initial candiates to evaluate, overwrites n_initial_samples

        verbose : bool, defualt = True
            Print out messages during optimization

        generator_options : dict
            Dictionary of options for MOBO

        Returns
        -------
        results : dict
            Dictionary with output data at the end of optimization

        """
    
    
    # Handle None
    generator_options = generator_options or {}

    generator_options.update({'fixed_cost': base_cost})
    generator = generators.multi_fidelity.MultiFidelityGenerator(vocs,
                                                                 **generator_options)

    return asynch_optimize(vocs,
                           evaluate_f,
                           generator,
                           budget,
                           processes,
                           base_cost,
                           output_path,
                           custom_model,
                           executor,
                           restart_file,
                           initial_x,
                           generator.tkwargs
                           )

