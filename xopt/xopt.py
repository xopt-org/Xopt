from .tools import full_path, expand_paths, load_config, save_config, fill_defaults, random_settings
from .cnsga import cnsga
from .configure import configure_algorithm, configure_simulation
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

# Algorithms
ALGORITHM_DEFAULTS= {
    'name':'cnsga',
    'function': 'xopt.cnsga.cnsga',
    'options':{
        'population':None,
        'max_generations':2,
        'population_size': 4,
        'crossover_probability':0.9,
        'mutation_probability':1.0,
        'selection':'auto',
        'verbose':True
    }
}    



SIMULATION_DEFAULTS = {
    'name':None,
    'evaluate':None,
    'options':None
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



ALL_DEFAULTS = {
    'xopt':XOPT_DEFAULTS,
    'algorithm':ALGORITHM_DEFAULTS,
    'simulation':SIMULATION_DEFAULTS,
    'vocs':VOCS_DEFAULTS
}




class Xopt:
    """
    
    input file should be a dict, JSON, or YAML file with top level keys
    
    xopt:
    
    algorithm:  
    
    simulation:
    
    vocs:
          
    """
    
    
    def __init__(self, config=None, verbose=True):
        

        # Internal state
        
        # Main configuration is in this nested dict
        self.config = None

        
        self.executor = None  
        self.verbose=verbose
        self.configured = False
                
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
        return self.config['algorithm']
    
    @property
    def vocs(self):
        return self.config['vocs']
    
    #--------------------------
    # Configure 
    
    def configure_xopt(self):   
        # Allows for .json or .yaml filenames as values. 
        self.config['xopt'] = load_config(self.config['xopt'])
        fill_defaults(self.config['xopt'], XOPT_DEFAULTS)

    def configure_algorithm(self):
        alg = configure_algorithm(self.config['algorithm'])
        
        # Register run function
        self.run_f = alg['function']
        options = alg['options']
        
        # Special
        if alg['name'] == 'cnsga':
            # Remove these from options. The class will provide these. 
            population = options.pop('population')
            self.population = load_config(population, self.verbose)
            for k in ['vocs', 'evaluate_f', 'output_path', 'toolbox']: # toolbox isn't needed
                options.pop(k)
                
        # update actual options
        self.config['algorithm'].update(options)
        

    def configure_simulation(self):
        self.simulation = configure_simulation(self.config['simulation'])                  
            
    def configure_vocs(self):
        # Allows for .json or .yaml filenames as values. 
        self.config['vocs'] = load_config(self.config['vocs'])
        fill_defaults(self.config['vocs'], VOCS_DEFAULTS)
        
        sim_name = self.vocs['simulation']
        if sim_name:
            assert sim_name == self.simulation['name'], f'VOCS simulation: {sim_name} has not been configured in xopt.'
        
        # Fill in these as options. TODO: Better logic?
        if self.vocs['templates']:
            self.simulation['options'].update(self.vocs['templates'])
                            
            
    def configure(self):
        """
        Configure everything
        
        Configuration order:
        xopt
        algorithm
        simulation
        vocs, which contains the simulation name, and templates
   
        """
        self.configure_xopt()
        self.configure_algorithm()
        self.configure_simulation()
        self.configure_vocs()
        
        # expand all paths
        self.config = expand_paths(self.config, ensure_exists=True)
   
        self.configured = True

    #--------------------------
    # Run
    
    def run(self, executor=None):
        alg = self.algorithm
        
        if alg['name'] == 'cnsga':
            self.run_cnsga(executor=executor)
        else:
            raise Exception(f'Unknown algorithm {alg}')
    
        
    def run_cnsga(self, executor=None):
        """Run the CNSGA algorithm with an executor"""
        assert self.configured, 'Not configured.'
        
        assert executor, 'Must provide an executor'
        alg = self.algorithm
        assert alg['name'] == 'cnsga'
        options = alg['options']
        output_path = self.config['xopt']['output_path']
        
        self.population = self.run_f(executor, vocs=self.vocs, population=self.population, evaluate_f=self.evaluate,
            output_path=output_path, **options)  

        
    #--------------------------
    # Evaluate    
         
    def random_inputs(self):
        return random_settings(self.vocs)
    
    def random_evaluate(self):
        inputs = self.random_inputs()
        return self.evaluate(inputs)
        
    def evaluate(self, inputs):
        """Evaluate should take one argument: A dict of inputs. """
        options = self.simulation['options']
        evaluate_f = self.simulation['evaluate_f']
        
        if options:
            return evaluate_f(inputs, **options)
        else:
            return evaluate_f(inputs)
    
    
    
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
    






