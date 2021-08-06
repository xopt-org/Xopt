from .tools import full_path, expand_paths, load_config, save_config, fill_defaults, random_settings, get_function, \
    isotime
from .cnsga import cnsga
from .sampler import random_sampler
from .configure import configure_algorithm, configure_simulation, configure_vocs, VOCS_DEFAULTS
from ._version import __version__
import pprint
from copy import deepcopy
import yaml
import json
import os

XOPT_DEFAULTS = {
    'output_path': '.',
    'verbose': True
}

SIMULATION_DEFAULTS = {
    'name': None,
    'evaluate': None,
    'options': None
}

# Algorithms
ALGORITHM_DEFAULTS = {
    'name': 'cnsga',
    'function': 'xopt.cnsga.cnsga',
    'options': {
        'population': None,
        'max_generations': 2,
        'population_size': 4,
        'crossover_probability': 0.9,
        'mutation_probability': 1.0,
        'selection': 'auto',
        'verbose': True
    }
}

ALL_DEFAULTS = {
    'xopt': XOPT_DEFAULTS,
    'algorithm': ALGORITHM_DEFAULTS,
    'simulation': SIMULATION_DEFAULTS,
    'vocs': VOCS_DEFAULTS
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
        self.verbose = verbose
        self.configured = False

        self.results = None

        if config:
            self.config = load_config(config, verbose=self.verbose)
            self.configure()

        else:
            # Make a template, so the user knows what is available
            self.vprint('Initializing with defaults')
            self.config = deepcopy(ALL_DEFAULTS)

    # --------------------------
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
    def simulation(self):
        return self.config['simulation']

    @property
    def vocs(self):
        return self.config['vocs']

    # --------------------------
    # Configure 

    def configure_xopt(self):
        # Allows for .json or .yaml filenames as values. 
        self.config['xopt'] = load_config(self.config['xopt'])

        fill_defaults(self.config['xopt'], XOPT_DEFAULTS)

    def configure_algorithm(self):
        alg = self.config['algorithm'] = configure_algorithm(self.config['algorithm'])
        options = alg['options']

        # Reserved keys
        for k in ['vocs', 'evaluate_f', 'output_path', 'toolbox']:
            if k in options:
                options.pop(k)

    def configure_simulation(self):
        self.config['simulation'] = configure_simulation(self.config['simulation'])

    def configure_vocs(self):
        self.config['vocs'] = configure_vocs(self.config['vocs'])

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

        # Get the actual functions
        self.run_f = get_function(self.algorithm['function'])
        self.evaluate_f = get_function(self.simulation['evaluate'])

        self.configured = True

    # --------------------------
    # Run

    def run(self, executor=None):
        assert self.configured, 'Not configured to run.'

        self.vprint(f'Starting at time {isotime()}')

        alg = self.algorithm['name']
        #self.algorithm['options']['executor'] = executor

        algorithms = ['random_sampler', 'bayesian_exploration', 'mobo']

        if alg == 'cnsga':
            self.run_cnsga(executor=executor)

        elif alg in algorithms:
            self.results = self.run_f(self.config,
                                      self.evaluate,
                                      output_path=self.config['xopt']['output_path'],
                                      **self.algorithm['options'])

        else:
            raise Exception(f'Unknown algorithm {alg}')

    def run_cnsga(self, executor=None):
        """Run the CNSGA algorithm with an executor"""
        options = self.algorithm['options']
        output_path = self.config['xopt']['output_path']

        # This takes priority
        # if self.population:
        #    options['population'] = self.population
        # elif 'population' not in options:
        #    print('Warning. Population not found in options.')
        #    options['population'] = None
        # else:
        #    options['population'] = load_config(options['population'], verbose=self.verbose)
        #
        ## Clean up
        # if options['population']:
        #    for k in list(options['population']):
        #        if k not in ['variables', 'generation']:
        #            print(f'removing {k}')
        #            options['population'].pop(k)

        self.population = self.run_f(executor=executor, vocs=self.vocs, evaluate_f=self.evaluate,
                                     output_path=output_path, **options)

        # --------------------------

    # Evaluate

    def random_inputs(self):
        return random_settings(self.vocs)

    def random_evaluate(self, check_vocs=True):
        """
        Makes random inputs and runs evaluate.
        
        If check_vocs, will check that all keys in vocs constraints and objectives are in output.
        """
        inputs = self.random_inputs()
        outputs = self.evaluate(inputs)
        if check_vocs:
            err_keys = []
            for k in self.vocs['objectives']:
                if k not in outputs:
                    err_keys.append(k)
            for k in self.vocs['constraints']:
                if k not in outputs:
                    err_keys.append(k)
            assert len(err_keys) == 0, f'Required keys not found in output: {err_keys}'

        return outputs

    def evaluate(self, inputs):
        """Evaluate should take one argument: A dict of inputs. """
        options = self.simulation['options']
        # evaluate_f = get_function(self.simulation['evaluate'])
        evaluate_f = self.evaluate_f
        if options:
            return evaluate_f(inputs, **options)
        else:
            return evaluate_f(inputs)

    # --------------------------
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
Version: {__version__}
Configured: {self.configured}
Config as YAML:
"""
        # return s+pprint.pformat(self.config)
        return s + yaml.dump(self.config, default_flow_style=None, sort_keys=False)

    def __str__(self):
        return self.__repr__()
