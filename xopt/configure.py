"""
Tools to configure an xopt run

"""
import xopt.bayesian_exploration
from xopt import tools

# -----------------------
# -----------------------
# Algorithm

KNOWN_ALGORITHMS = {
    'cnsga': 'xopt.cnsga.cnsga',
    'random_sampler': 'xopt.sampler.random_sampler',
    'bayesian_exploration': 'xopt.bayesian_exploration.bayesian_exploration'
}

ALGORITHM_DEFAULTS = {
    'name': 'cnsga',
    'function': 'xopt.cnsga.cnsga',
    'options': {}
}


def configure_algorithm(config):
    """
    Configures a algorithm config dict. The dict should have:
    
    'name': <string that VOCS refers to>
    'function': <fully qualified function name>
    'options': <any options. Default is empty {}> 
    
    Example:
        
    """

    for k in config:
        if k not in ALGORITHM_DEFAULTS:
            raise ValueError(f'unknown algoritm key: {k}, allowed: {list(ALGORITHM_DEFAULTS)}')

    name = config['name']  # required

    if 'function' not in config or not config['function']:
        if name in KNOWN_ALGORITHMS:
            f_name = KNOWN_ALGORITHMS[name]
        else:
            raise ValueError(f'Algorthm {name} must provide its fully qualified function name.')
    else:
        f_name = config['function']

    # Make sure this works, and get the options. This
    f = tools.get_function(f_name)
    options = {}
    if 'options' in config:
        options.update(config['options'])
    defaults = tools.get_function_defaults(f)
    tools.fill_defaults(options, defaults, strict=False)

    return {'name': config['name'], 'function': f_name, 'options': options}


# -----------------------
# -----------------------
# Simulation

def configure_simulation(config):
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

    name = config['name']  # required

    f_name = config['evaluate']

    if f_name:
        f = tools.get_function(f_name)
    else:
        f = None

    if 'options' in config:
        options = config['options']
    else:
        options = {}

    n_required_args = tools.get_n_required_fuction_arguments(f)
    assert n_required_args == 1, f'{name} has {n_required_args}, but should have exactly one.'

    defaults = tools.get_function_defaults(f)

    tools.fill_defaults(options, defaults, strict=False)

    return {'name': name, 'evaluate': f_name, 'options': options}


# -----------------------
# -----------------------
# VOCS

VOCS_DEFAULTS = {
    'name': None,
    'description': None,
    'simulation': None,
    'templates': None,
    'variables': None,
    'objectives': None,
    'constraints': {},
    'linked_variables': {},
    'constants': {}
}


def configure_vocs(config):
    # Allows for .json or .yaml filenames as values.
    vocs = tools.load_config(config)
    tools.fill_defaults(vocs, VOCS_DEFAULTS)
    return vocs
