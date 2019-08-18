"""
Tools to configure an xopt run

"""

from xopt.tools import load_vocs, full_path, add_to_path
#from xopt.nsga2_tools import nsga2_toolbox

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

[sampler_config]
chunk_size: 100
max_samples: 1000000

"""



def load_config(filePath):
    """
    Load INI style config file for xopt and nsga2. 
    
    Will add defaults. Otherewise, there is no processing. 
    
    Returns a dict.
    
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
    
    

    #---------------------------      
    # astra_config
    if 'astra_config' in config:
        c = config['astra_config']
        d['astra_config'] = {'timeout':c.getint('timeout', None),
                           'workdir':c.get('workdir', '/tmp'),    # Workdir must be set. Otherswise, work will be done in template!
                           'verbose':c.getint('workdir', 0),
                           'astra_bin':c.get('astra_bin', '$ASTRA_BIN'),
                           'generator_bin':c.get('generator_bin', '$GENERATOR_BIN')
                             
        }    
    
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
            'checkpoint':c.get('checkpoint', ''),
            'checkpoint_frequency':c.getint('checkpoint_frequency', 1),
            'verbose':c.getboolean('verbose', True)
        }
    
    #---------------------------      
    # sampler_config
    if 'sampler_config' in config:
        c = config['sampler_config']
        d['sampler_config'] = {
            'chunk_size':c.getint('chunk_size', 100),
            'max_samples':c.getint('max_samples', 100000000)
        }    
    
        
    return d
#c1 = load_config('xopt.in')
    
    
    
def configure(config):
    """"
    Processes config dict, adding vocs and full paths. 
    """
    
    c = {}
    c.update(config)
    
    # xopt_config
    c['xopt_config']['vocs_file'] = full_path(config['xopt_config']['vocs_file'])
    c['xopt_config']['output_dir'] = full_path(config['xopt_config']['output_dir'])
    
    # vocs
    c['vocs'] = load_vocs( config['xopt_config']['vocs_file'])    
    
    # nsga2
    if 'nsga2_config' in c:
        if c['nsga2_config']['checkpoint'] != '':
            c['nsga2_config']['checkpoint'] = full_path(c['nsga2_config']['checkpoint'])
    
    # Options for templates
    if 'template_dir' in c['vocs']:
        c['vocs']['template_dir'] =  full_path(c['vocs']['template_dir'])
    if 'templates' in c['vocs']:
        for k in c['vocs']['templates']:
            c['vocs']['templates'][k] = full_path(c['vocs']['templates'][k])
    
    
    # Binary paths
    # Prepend to $PATH
    if 'gpt_config' in c:
        c['gpt_config']['gpt_path'] = add_to_path( config['gpt_config']['gpt_path'] )
    if 'distgen_config' in c:
        c['distgen_config']['distgen_path'] =  add_to_path(config['distgen_config']['distgen_path'])
     
    # Executables 
    if 'astra_config' in c:
        c['astra_config']['astra_bin'] = full_path(config['astra_config']['astra_bin'])
        c['astra_config']['generator_bin'] = full_path(config['astra_config']['generator_bin'])
    
    return c
    
    
