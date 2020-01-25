"""
Tools to configure an xopt run

"""

from xopt import tools


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
    print(config)
    f_name = config['evaluate']
    f = tools.get_function(f_name)
    
    if 'options' in config:
        options = config['options']
    else:
        options = {}
        
    n_required_args = tools.get_n_required_fuction_arguments(f)
    assert n_required_args == 1, f'{name} has {n_required_args}, but should have exactly one.'
    
    defaults = tools.get_function_defaults(f)

    tools.fill_defaults(options, defaults, strict=False)
    
    return {'name':config['name'], 'evaluate_f':f, 'options':options}
    

    
