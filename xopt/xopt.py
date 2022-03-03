

from copy import deepcopy

import yaml

from . import configure
from xopt.legacy import reformat_config
from xopt import __version__
from xopt.tools import expand_paths, load_config, save_config,\
    random_settings, get_function, isotime

import logging
logger = logging.getLogger(__name__)

import sys


class Xopt:
    """
    
    Object to handle a single optimization problem.
    
    Parameters
    ----------
    config: dict, YAML text, JSON text
        input file should be a dict, JSON, or YAML file with top level keys
    
          
    """

    def __init__(self, config=None):

        # Internal state

        # Main configuration is in this nested dict
        self.config = deepcopy(config)
        self.configured = False

        self.results = None

        self.run_f = None
        self.evaluate_f = None

        if config:
            self.config = load_config(self.config)

            # make sure configure has the required keys
            for name in configure.ALL_DEFAULTS:
                if name not in self.config:
                    raise Exception(f'Key {name} is required in config for Xopt')

            # load any high level config files
            for ele in ['xopt', 'simulation', 'algorithm', 'vocs']:
                self.config[ele] = load_config(self.config[ele])

            # reformat old config files if needed
            self.config = reformat_config(self.config)

            # do configuration
            self.configure_all()

        else:
            # Make a template, so the user knows what is available
            logger.info('Initializing with defaults')
            self.config = deepcopy(configure.ALL_DEFAULTS)

    def configure_all(self):
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
    # Configure
    def configure_xopt(self):
        """ configure xopt """
        # check and fill defaults
        configure.configure_xopt(self.config['xopt'])

    def configure_algorithm(self):
        """ configure algorithm """
        self.config['algorithm'] = configure.configure_algorithm(self.config[
                                                                    'algorithm'])

    def configure_simulation(self):
        self.config['simulation'] = configure.configure_simulation(self.config[
                                                                    'simulation'])

    def configure_vocs(self):
        self.config['vocs'] = configure.configure_vocs(self.config['vocs'])


    # --------------------------
    # Saving and Loading from file
    def load(self, config):
        """Load config from file (JSON or YAML) or data"""
        self.config = load_config(config)
        self.configure_all()

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
    # Run

    def run(self, executor=None):
        assert self.configured, 'Not configured to run.'

        logger.info(f'Starting at time {isotime()}')

        opts = self.algorithm['options']

        # Special for genetic algorithms
        if self.results and 'population' in opts:
            opts['population'] = self.results

        self.results = self.run_f(vocs=self.vocs,
                                  evaluate_f=self.evaluate,
                                  executor=executor,
                                  output_path=self.config['xopt']['output_path'],
                                  **opts)

    def random_inputs(self):
        return random_settings(self.vocs)

    def random_evaluate(self, check_vocs=True):
        """
        Makes random inputs and runs evaluate.
        
        If check_vocs, will check that all keys in vocs constraints and objectives
        are in output.
        """
        inputs = self.random_inputs()
        outputs = self.evaluate(inputs)
        if check_vocs:
            err_keys = []
            for k in self.vocs.objectives:
                if k not in outputs:
                    err_keys.append(k)
            for k in self.vocs.constraints:
                if k not in outputs:
                    err_keys.append(k)
            assert len(err_keys) == 0, f'Required keys not found in output: {err_keys}'

        return outputs

    def evaluate(self, inputs):
        """Evaluate should take one argument: A dict of inputs. """
        options = self.simulation['options']
        evaluate_f = self.evaluate_f
        if options:
            return evaluate_f(inputs, **options)
        else:
            return evaluate_f(inputs)

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
        # Cast to dicts for nice printout
        config = {k:dict(v) for k, v in  self.config.items()}
        
        return s + yaml.dump(config, default_flow_style=None,
                             sort_keys=False)

    def __str__(self):
        return self.__repr__()
