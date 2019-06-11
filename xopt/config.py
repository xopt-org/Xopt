"""
Tools to configure an xopt run

"""

from xopt import legacy
from xopt.tools import load_vocs
from xopt.nsga2_tools import nsga2_toolbox

from configparser import ConfigParser
import os









XOPT_CONFIG_DEFAULTS = {
    'checkpoint'           : '',
    'max_generations'      : '2',
    'population_size'      : '100',
    'checkpoint_frequency' : '1',
    'do_archive'           : 'false',
    'abort_file'           : '__jbooty' # Named for historical reasons
                     }
def load_config(filePath):
    """
    Load INI style config file for xopt and nsga2. 
    
    """
    config = ConfigParser()
    config['DEFAULT'] =  XOPT_CONFIG_DEFAULTS
    assert os.path.exists(filePath), 'xopt input file does not exist: '+filePath
    config.read(filePath)
    
    d = {}
    #---------------------------
    # xopt_config (required)
    xopt = config['xopt_config']
    d['xopt_config'] = dict(xopt)
    # Bools
    for k in ['do_archive', 'skip_checkpoint_eval']:
        d['xopt_config'][k]  = xopt.getboolean(k)
        
    #---------------------------      
    # gpt_config
    if 'gpt_config' in config:
        gpt = config['gpt_config']
        d['gpt_config'] = {'timeout':gpt.getint('timeout', None)
        }
    
    #---------------------------       
    # nsga2_config   
    d['nsga2_config'] = dict(config['nsga2_config'])
    # ints
    for k in ['population_size', 'max_generations', 'checkpoint_frequency']:
        d['nsga2_config'][k] = config.getint('nsga2_config', k)
        
    return d
    
    
    
def configure(config):
    vocs = load_vocs(config['xopt_config']['vocs_file'])
    
    # Create the toolbox
    toolbox_params = legacy.toolbox_params(variable_dict=vocs['variables'], constraint_dict = vocs['constraints'], objective_dict = vocs['objectives'])
    toolbox = nsga2_toolbox(**toolbox_params)
    
    
    return {'toolbox':toolbox, 'vocs':vocs}    