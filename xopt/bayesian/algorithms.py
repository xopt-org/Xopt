from typing import Dict
from xopt.tools import get_function
from . import generators
from .optimize import optimize
from .models.models import create_multi_fidelity_model
from xopt.vocs import VOCS

KWARG_DEFAULTS = {
    "output_path": None,
    "custom_model": None,
    "executor": None,
    "restart_file": None,
    "initial_x": None,
    "generator_options": {},
}


def bayesian_optimize(vocs, evaluate_f, n_steps=1, n_initial_samples=1, **kwargs):
    """
    Bayesian optimization

    Bayesian optimization with an arbitrary acquisition function. Required to pass a
    generator_options dict with the keyword `acquisition_function` that points to a
    valid botorch.acquisition.AcquisitionFunction object or a callable that returns
    an object of that type, see examples/bayes_opt.

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

    Returns
    -------
    results : dict
        Dictionary with output data at the end of optimization

    Other Parameters
    ----------------
    **kwargs : `~xopt.bayesian.optimizer` properties
    """

    options = KWARG_DEFAULTS.copy()
    options.update(kwargs)

    generator_options = options.pop("generator_options")

    try:
        # Required
        acq_func = get_function(generator_options.pop("acquisition_function", None))
    except ValueError:
        raise ValueError(
            "acquisition_function is a required parameter of generator_options."
        )

    generator = generators.generator.BayesianGenerator(
        vocs, acq_func, **generator_options
    )
    return optimize(
        vocs,
        evaluate_f,
        generator,
        n_steps,
        n_initial_samples,
        tkwargs=generator.tkwargs,
        **options,
    )


def mobo(vocs, evaluate_f, ref=None, n_steps=1, n_initial_samples=1, **kwargs):
    """
    Multi-objective Bayesian optimization

    This algorithm attempts to determine the Pareto front of a multi-objective
    problem using multi-objective Bayesian optimization. The acquisition function is
    the expected hypervolume improvement (EHVI), implemented in botorch. Allows for
    the specification of unknown constraints and proximal biasing.

    See https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.24.062801
    and https://botorch.org/tutorials/multi_objective_bo for details.

    Note: while this algorithm can be used with parallel evaluations, optimization of
    the acquistion function can become prohibitively expensive given the number of
    objectives, points on the Pareto front and number of parallelized evaluations.

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

    Returns
    -------
    results : dict
        Dictionary with output data at the end of optimization

    Other Parameters
    ----------------
    **kwargs : `~xopt.bayesian.optimizer` properties

    """

    options = KWARG_DEFAULTS.copy()
    options.update(kwargs)

    generator_options = options.pop("generator_options")

    vocs = VOCS.parse_obj(vocs)

    generator = generators.mobo.MOBOGenerator(vocs, ref, **generator_options)

    return optimize(
        vocs,
        evaluate_f,
        generator,
        n_steps,
        n_initial_samples,
        tkwargs=generator.tkwargs,
        **options,
    )


def bayesian_exploration(vocs, evaluate_f, n_steps=1, n_initial_samples=1, **kwargs):
    """
    Bayesian Exploration

    This optimization algorithm attempts to efficiently characterize a target
    function by sampling points that maximize model uncertainty. By combining this
    acquisition function with a Gaussian process model that uses automatic relevance
    determination (ARD) we increase the number of samples along the fastest changing
    axis of the function. Natively incorperates inequality constraints.

    See https://www.nature.com/articles/s41467-021-25757-3 for detials.

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

    Returns
    -------
    results : dict
        Dictionary with output data at the end of optimization

    Other Parameters
    ----------------
    **kwargs : `~xopt.bayesian.optimizer` properties

    """

    options = KWARG_DEFAULTS.copy()
    options.update(kwargs)

    generator_options = options.pop("generator_options")

    generator = generators.exploration.BayesianExplorationGenerator(
        vocs, **generator_options
    )
    return optimize(
        vocs,
        evaluate_f,
        generator,
        n_steps=n_steps,
        n_initial_samples=n_initial_samples,
        tkwargs=generator.tkwargs,
        **options,
    )


def multi_fidelity_optimize(
    vocs, evaluate_f, budget=1, processes=1, base_cost=1.0, **kwargs
):
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

    Returns
    -------
    results : dict
        Dictionary with output data at the end of optimization

    Other Parameters
    ----------------
    **kwargs : `~xopt.bayesian.optimizer` properties

    """
    options = KWARG_DEFAULTS.copy()
    options.update(kwargs)

    generator_options = options.pop("generator_options")

    if options["custom_model"] is None:
        options["custom_model"] = create_multi_fidelity_model

    generator_options.update({"fixed_cost": base_cost})
    generator = generators.multi_fidelity.MultiFidelityGenerator(
        vocs, **generator_options
    )

    return optimize(
        vocs,
        evaluate_f,
        generator,
        budget=budget,
        processes=processes,
        base_cost=base_cost,
        tkwargs=generator.tkwargs,
        **options,
    )

