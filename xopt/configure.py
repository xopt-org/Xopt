"""
Tools to configure an xopt run

"""
from copy import deepcopy

from xopt import tools
from xopt.vocs import VOCS
import logging
from typing import Dict

logger = logging.getLogger(__name__)

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
    'output_path': '.'
}

SIMULATION_DEFAULTS = {
    'name': None,
    'evaluate': None,
    'options': None,
}

# Algorithms
ALGORITHM_DEFAULTS = {
    'name': None,
    'function': None,
    'options': None,
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


def configure_xopt(xopt_config: Dict) -> None:
    check_config_against_defaults(xopt_config, XOPT_DEFAULTS)
    xopt_config = fill_defaults(xopt_config, XOPT_DEFAULTS)
    return xopt_config


def configure_algorithm(alg_config: Dict) -> Dict:
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

    # Reserved keys
    for k in ['vocs', 'executor', 'evaluate_f', 'output_path', 'toolbox']:
        if k in options:
            options.pop(k)

            # update alg_config with full_options
    alg_config['options'] = options
    return alg_config


# -----------------------
# -----------------------
# Simulation

def configure_simulation(sim_config: Dict) -> Dict:
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
    check_config_against_defaults(sim_config, SIMULATION_DEFAULTS)
    fill_defaults(sim_config, SIMULATION_DEFAULTS)
    
    f_name = sim_config['evaluate']
    
    f = tools.get_function(f_name)

    options = sim_config['options'] or {}

    n_required_args = tools.get_n_required_fuction_arguments(f)
    assert n_required_args == 1, f'function has {n_required_args}, but should have ' \
                                 f'exactly one. '

    defaults = tools.get_function_defaults(f)

    fill_defaults(options, defaults)

    sim_config.update({'evaluate': f_name, 'options': options})

    return sim_config


# -----------------------
# -----------------------
# VOCS


def configure_vocs(vocs_config):
    # Allows for .json or .yaml filenames as values.
    #check_config_against_defaults(vocs_config, VOCS_DEFAULTS)
    #fill_defaults(vocs_config, VOCS_DEFAULTS)

    return VOCS.parse_obj(vocs_config)


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

    return dict1


def check_config_against_defaults(test_dict, defaults):
    for k in test_dict:
        if k not in defaults:
            raise Exception(
                f'Extraneous key: {k}. Allowable keys: ' + ', '.join(list(defaults)))
