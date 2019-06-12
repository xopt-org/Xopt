"""
Tools to configure an xopt run

"""

from xopt.tools import load_vocs
from xopt.nsga2_tools import nsga2_toolbox

from configparser import ConfigParser
import os


XOPT_EXAMPLE_CONFIGFILE = """
Load INI style config file for xopt. 


Example: 

[xopt_config]
vocs_file            : cbetaDL_measurement.json
abort_file           :  __jbooty
skip_checkpoint_eval :True 
checkpoint : pop_432.pkl
do_archive  :True

[gpt_config]
timeout: 400
gpt_path: /global/homes/c/cmayes/GitHub/xgpt/gpt321-rhel7/bin/

[distgen_config]
distgen_path: /global/homes/c/cmayes/cori/distgen/bin/

[nsga2_config]
population_size: 60

"""



XOPT_CONFIG_DEFAULTS = {
    'checkpoint'           : '',
    'max_generations'      : '2',
    'population_size'      : '100',
    'checkpoint_frequency' : '1',
    'do_archive'           : 'false',
    'output_dir'           : '.',
    'abort_file'           : '__jbooty' # Named for historical reasons
                     }


def load_config(filePath):
    """
    Load INI style config file for xopt. 
    
    Does no processing (doens't load vocs_file, etc.). 
    
    Returns a dict. 
    
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
        c = config['gpt_config']
        d['gpt_config'] = {'timeout':c.getint('timeout', None),
                           'workdir':c.get('workdir', None),
                           'verbose':c.getint('workdir', 0),
                           'gpt_path':c['gpt_path']                  
        }
        
    #---------------------------      
    # distgen_config
    if 'distgen_config' in config:
        c = config['distgen_config']
        d['disgtgen_config'] = {'distgen_path':c['distgen_path']}   
    
    #---------------------------       
    # nsga2_config   
    d['nsga2_config'] = dict(config['nsga2_config'])
    # ints
    for k in ['population_size', 'max_generations', 'checkpoint_frequency']:
        d['nsga2_config'][k] = config.getint('nsga2_config', k)
        
    return d
#c1 = load_config('xopt.in')
    
    
