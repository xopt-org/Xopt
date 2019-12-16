from .tools import full_path, expand_paths, load_config, save_config, fill_defaults, random_settings
from .cnsga import cnsga
import pprint
from copy import deepcopy
import yaml
import json
import os


XOPT_DEFAULTS = {
    'output_path':'.',
    'verbose':True,
    'algorithm':'cnsga'
}

VOCS_DEFAULTS = {
    'name':None,
    'description':None,
    'simulation':None,
    'templates':None,
    'variables':None,
    'objectives':None,
    'constraints':None,
    'linked_variables':None,
    'constants':None
}

# Simulation goes here
# astra_with_generator = {}



# Algorithms
CNSGA_DEFAULTS= {
    'max_generations':2,
    'population_size': 4,
    'crossover_probability':0.9,
    'mutation_probability':1.0,
    'selection':'auto',
    'verbose':True
}    

ALL_DEFAULTS = {
    'xopt':XOPT_DEFAULTS,
    'cnsga':CNSGA_DEFAULTS,
    'vocs':VOCS_DEFAULTS
    
}





class Xopt:
    """
    
    input file should be a dict, JSON, or YAML file with top level keys
    
    xopt:
    
    vocs:
        
        
    [algorithm]:    
        
    [simulation]:
    
    """
    
    
    def __init__(self, config=None, verbose=True):
        

        # Internal state
        
        # Main configuration is in this nested dict
        self.config = None

        
        self.executor = None  
        self.verbose=verbose
        self.configured = False
        
        # These will need to be set to run an optimization
        self.sim_evaluate = None
        self.evaluate_options = None
        
        if config:
            self.config = load_config(config, verbose=self.verbose)  
            self.configure()
        else:
            # Make a template, so the user knows what is available
            self.vprint('Initializing with defaults')
            self.config = deepcopy(ALL_DEFAULTS)

    #--------------------------
    # Saving and Loading from file
    
    def load(self, config):
        """Load config from file (JSON or YAML) or data"""
        self.config = load_config(config)
        
    def save(self, file):
        """Save config to file (JSON or YAML)"""
        save_config(self.config, file)
                
    
    # Conveniences
    @property
    def algorithm(self):
        return self.config['xopt']['algorithm']
    @property
    def vocs(self):
        return self.config['vocs']
    @property
    def simulation(self):
        return self.config['vocs']['simulation']  
    
    
    
    #--------------------------
    # Configure 
    
    def configure_xopt(self):   
        # Allows for .json or .yaml filenames as values. 
        self.config['xopt'] = load_config(self.config['xopt'])
        fill_defaults(self.config['xopt'], XOPT_DEFAULTS)

    def configure_vocs(self):
        # Allows for .json or .yaml filenames as values. 
        self.config['vocs'] = load_config(self.config['vocs'])
        fill_defaults(self.config['vocs'], VOCS_DEFAULTS)
        
             
    def configure_simulation(self, simulation=None):

        if simulation:
            sim = simulation
        else:
            sim = self.simulation
        
        if not sim:
            print('no simulation to configure')
            self.configured=False
            return

        
        #--------------------------
        # astra, astra_with_generator, astra_with_distgen
        if sim in ['astra', 'astra_with_generator', 'astra_with_distgen']:
            configure_astra(self, sim)    
            
        elif sim ==  'test_TNK':
            from xopt.evaluators import test_TNK
            self.sim_evaluate = test_TNK.evaluate_TNK
            
        else:
            raise Exception(f'unknown simulation {sim}')
        
        self.vprint(f'Simulation {sim} configured')
        self.configured = True
        
    def configure_algorithm(self):
        alg = self.algorithm
        if alg == 'cnsga':
            self.population = None
            fill_defaults(self.config['cnsga'], CNSGA_DEFAULTS)
            
    def configure(self):
        """
        Configure everything
        
        Configuration order:
        xopt
        algorithm
        vocs, which contains the simulation string
        [simulation]
       
        
        
        """
        self.configure_xopt()
        self.configure_algorithm()
        self.configure_vocs()
        if self.simulation:
            self.configure_simulation()
            
        else:
            self.vprint('Warning: no simulation specified. Not configured')
            self.configured = False
        
        # expand all paths
        self.config = expand_paths(self.config, ensure_exists=True)
   

    #--------------------------
    # Run
        
        
    def run_cnsga(self, executor=None):
        """Run the CNSGA algorithm with an executor"""
        assert self.configured, 'Not configured.'
        
        assert executor, 'Must provide an executor'
        
        cnsga_config = self.config['cnsga']
        output_path = self.config['xopt']['output_path']
        
        self.population = cnsga(executor, vocs=self.vocs, population=self.population, evaluate_f=self.evaluate,
            output_path=output_path, **cnsga_config)  

        
    #--------------------------
    # Evaluate    
         
    def random_inputs(self):
        return random_settings(self.vocs)
    
    def random_evaluate(self):
        inputs = self.random_inputs()
        return self.evaluate(inputs)
        
    def evaluate(self, inputs):
        """Evaluate should take one argument: A dict of inputs. """
        options = self.evaluate_options
        if options:
            return self.sim_evaluate(inputs, **options)
        else:
            return self.sim_evaluate(inputs)
    
    
    
    #--------------------------
    # Helpers and utils
    
    def vprint(self, *args, **kwargs):
        """Verbose print"""
        if self.verbose:
            print(*args, **kwargs)
    
    
    def __getitem__(self, config_item):
        """
        Get a configuration attribute
        """
        return self.config[config_item]
    
    def __repr__(self):
        s = f"""
            Xopt
________________________________           

Configured: {self.configured}

Config as YAML:
"""
        #return s+pprint.pformat(self.config)
        return s + yaml.dump(self.config, default_flow_style=None, sort_keys=False)
    
    def __str__(self):
        s = f''
        return s
    
# Custom configs   
    
#--------------------------
# astra, astra_with_generator, astra_with_distgen

def configure_astra(xopt, sim):    
    """
    Configures astra for an xopt instance.
    
    """
    # Use lume-astra function directly
    from astra.evaluate import configure_astra_evaluate
    
    if sim not in xopt.config:
        xopt.config[sim] = {} # TEST
     
    # Allows for .json or .yaml filenames as value.
    xopt.config[sim] = load_config(xopt.config[sim])
    
    # Add template input files
    if xopt.vocs['templates']:
        xopt.config[sim].update(xopt.vocs['templates'])                
    
    d = configure_astra_evaluate(simulation=sim, config=xopt.config[sim])
    fill_defaults(xopt.config[sim], d['options'])

    xopt.config[sim] = expand_paths(xopt.config[sim])
    
    xopt.sim_evaluate = d['evaluate_f']
    xopt.evaluate_options = d['options']    
    



