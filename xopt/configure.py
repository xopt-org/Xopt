"""
Tools to configure an xopt run

"""
from copy import deepcopy

from xopt import tools
import logging
from typing import Dict

# -----------------------
# -----------------------
# Algorithm

KNOWN_ALGORITHMS = {
    'cnsga': 'xopt.cnsga.cnsga',
    'random_sampler': 'xopt.sampler.random_sampler',
    'bayesian_optimization': 'xopt.bayesian.algorithms.bayesian_optimize',
    'bayesian_exploration': 'xopt.bayesian.algorithms.bayesian_exploration',
    'mobo': 'xopt.bayesian.algorithms.mobo',
    'multi_fidelity': 'xopt.bayesian.algorithms.multi_fidelity_optimize'
}

# defaults for required dict keys
XOPT_DEFAULTS = {
    'output_path': '.',
    'logging': logging.WARNING,
}

SIMULATION_DEFAULTS = {
    'name': None,
    'evaluate': None,
    'options': None,
    'templates': None,
    'logging': logging.WARNING,
}

# Algorithms
ALGORITHM_DEFAULTS = {
    'name': None,
    'function': None,
    'options': None,
    'logging': logging.WARNING,
}

VOCS_DEFAULTS = {
    'variables': None,
    'objectives': None,
    'constraints': None,
    'linked_variables': None,
    'constants': None
}

ALL_DEFAULTS = {
    'xopt': XOPT_DEFAULTS,
    'algorithm': ALGORITHM_DEFAULTS,
    'simulation': SIMULATION_DEFAULTS,
    'vocs': VOCS_DEFAULTS
}

logger = logging.getLogger(__name__)


def configure_xopt(xopt_config: Dict) -> None:
    check_config_against_defaults(xopt_config, XOPT_DEFAULTS)
    fill_defaults(xopt_config, XOPT_DEFAULTS)


def configure_algorithm(alg_config: Dict) -> None:
    """
    Configures a algorithm config dict. The dict should have:
    
    'name': <name of algorithm>
    'function': <fully qualified function name>
    'options': <any options. Default is empty {}>
    
    Example:
        
    """
    check_config_against_defaults(alg_config, ALGORITHM_DEFAULTS)
    fill_defaults(alg_config, ALGORITHM_DEFAULTS)

    # check if EITHER name is valid known algorithm OR function is not None
    if alg_config['name'] in KNOWN_ALGORITHMS or alg_config['function'] is not None:
        # if BOTH known algorithm name and function is specified raise warning - use
        # known algorithm
        if (alg_config['name'] in KNOWN_ALGORITHMS and
                alg_config['function'] is not None):
            logger.warning(f'Specified both known algorithm `{alg_config["name"]}` and '
                           f'`function`. Using known algorithm function.')

            # populate function with known algorithm
            alg_config['function'] = KNOWN_ALGORITHMS[alg_config['name']]
        # if only named algorithm is specified populate function with corresponding
        # function
        elif alg_config['function'] is None:
            alg_config['function'] = KNOWN_ALGORITHMS[alg_config['name']]

    else:
        raise ValueError(f'unknown algoritm key and no algorithm function specified')

    # get function for inspection and make sure the function is callable
    f = tools.get_function(alg_config['function'])

    # add default arguments from function to options dict
    options = {}
    if 'options' in alg_config:
        options.update(alg_config['options'])
    defaults = tools.get_function_defaults(f)
    fill_defaults(options, defaults)

    # see if generator_options are in options - functions can be specified in
    # generator options
    if 'generator_options' in options:
        if options['generator_options'] is not None:
            for name, val in options['generator_options'].items():
                if isinstance(val, str):
                    options['generator_options'][name] = tools.get_function(val)

    # update alg_config with full_options
    alg_config.update(options)


# -----------------------
# -----------------------
# Simulation

def configure_simulation(sim_config):
    """
    Configures a simulation config dict. The dict should have:
    
    'name': <string that VOCS refers to>
    'evaluate': <fully qualified function name>
    'options': <any options. Default is empty {}> 
    
    Example:
    
     {'name': 'astra_with_generator',
     'evaluate': 'astra.evaluate.evaluate_astra_with_generator',
     'options': {'archive_path': '.', 'merit_f': None}}
        
    """

    name = sim_config['name']  # required

    f_name = sim_config['evaluate']

    if f_name:
        f = tools.get_function(f_name)
    else:
        f = None

    if 'options' in sim_config:
        options = sim_config['options']
    else:
        options = {}

    n_required_args = tools.get_n_required_fuction_arguments(f)
    assert n_required_args == 1, f'{name} has {n_required_args}, but should have exactly one.'

    defaults = tools.get_function_defaults(f)

    fill_defaults(options, defaults)

    sim_config.update({'name': name, 'evaluate': f_name, 'options': options})


# -----------------------
# -----------------------
# VOCS


def configure_vocs(vocs_config):
    # Allows for .json or .yaml filenames as values.
    vocs_config = tools.load_config(vocs_config)
    fill_defaults(vocs_config, VOCS_DEFAULTS)

    for key in vocs_config:
        if vocs_config[key] == {}:
            vocs_config[key] = None


# --------------------------------
# adding defaults to dicts
def fill_defaults(dict1, defaults):
    """
    Fills a dict with defaults in a defaults dict.

    dict1 must only contain keys in defaults.

    deepcopy is necessary!

    """
    for k, v in defaults.items():
        if k not in dict1:
            dict1[k] = deepcopy(v)


def check_config_against_defaults(test_dict, defaults):
    if 'verbose' in test_dict:
        print('WARNING: verbose keyword is depreciated, use `logging` instead.')
        verbose = test_dict.pop('verbose')

        test_dict['logging'] = logging.WARNING
        if verbose:
            test_dict['logging'] = logging.INFO

    for k in test_dict:
        if k not in defaults:
            raise Exception(
                f'Extraneous key: {k}. Allowable keys: ' + ', '.join(list(defaults)))
