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



def load_config(filePath):
    """
    Load INI style config file for xopt and nsga2. 
    
    """
    config = ConfigParser()
    ##config['DEFAULT'] =  XOPT_CONFIG_DEFAULTS
    assert os.path.exists(filePath), 'xopt input file does not exist: '+filePath
    config.read(filePath)
    
    d = {}
    #---------------------------
    # xopt_config (required)
    c = config['xopt_config']
    d['xopt_config'] = {
    'vocs_file':c.get('vocs_file', 'vocs.json'),
    'output_dir':c.get('output_dir', '.')
    }
    print(d['xopt_config'])
    
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
        d['distgen_config'] = {'distgen_path':c['distgen_path']}   
    
    #---------------------------       
    # nsga2_config
    if 'nsga2_config' in config:
        c = config['nsga2_config']
        d['nsga2_config'] = {
            'do_archive':c.getboolean('do_archive', True),
            'skip_checkpoint_eval':c.getboolean('skip_checkpoint_eval', False),
            'population_size':c.getint('population_size', 0), # 0 is intended to be automatiaclly adjusted. 
            'max_generations':c.getint('max_generations', 100),
            'checkpoint_frequency':c.getint('checkpoint_frequency', 1)
        }
    
        
    return d
#c1 = load_config('xopt.in')
    
    
