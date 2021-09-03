from . import generators
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

    assert (isinstance(generator_options, dict) and
            'acquisition_function' in generator_options)
    acq_func = generator_options.pop('acquisition_function')
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

    verbose : bool, defualt = True
        Print out messages during optimization

    generator_options : dict
        Dictionary of options for MOBO

    Returns
    -------
    results : dict
        Dictionary with output data at the end of optimization

    """

    if generator_options is None:
        generator_options = {}

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
    if generator_options is None:
        generator_options = {}

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
                            n_steps=1,
                            n_initial_samples=1,
                            output_path=None,
                            custom_model=create_multi_fidelity_model,
                            executor=None,
                            restart_file=None,
                            initial_x=None,
                            verbose=True,
                            generator_options=None):
    """
        Multi-fidelity optimization

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
    if generator_options is None:
        generator_options = {}

    generator = generators.multi_fidelity.MultiFidelityGenerator(vocs,
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
