import matplotlib.pyplot as plt
import numpy as np
import torch
import logging

from xopt import Xopt
from xopt.bayesian.generators.multi_fidelity import MultiFidelityGenerator
from xopt.bayesian.models.models import create_multi_fidelity_model
from xopt.evaluators import test_3d, test_1d
from concurrent.futures import ThreadPoolExecutor
import copy


class TestMultiFidelity:
    VOCS = test_3d.VOCS
    config = {'vocs': test_3d.VOCS.copy()}
    config['simulation'] = {'name': 'AugmentedHartmann',
                            'evaluate': 'xopt.evaluators.test_3d.evaluate'}
    config['xopt'] = {'output_path': '', 'verbose': True,
                      'algorithm': 'multi_fidelity'}
    config['algorithm'] = {'name': 'multi_fidelity',
                           'options': {
                               'processes': 4,
                               'budget': 5,
                               'base_cost': 1.0,
                               'generator_options': {}
                           }}

    config['algorithm']['options']['generator_options']['num_restarts'] = 2
    config['algorithm']['options']['generator_options']['raw_samples'] = 2
    config['algorithm']['options']['generator_options']['base_acq'] = None

    def test_multi_fidelity_base(self):
        test_config = copy.deepcopy(self.config)
        X = Xopt(test_config)
        executor = ThreadPoolExecutor()
        X.run(executor=executor)

    def test_multi_fidelity_restart_file(self):
        test_config = copy.deepcopy(self.config)
        test_config['algorithm']['options']['restart_file'] = 'asynch_test_results.json'
        X = Xopt(test_config)
        executor = ThreadPoolExecutor()
        X.run(executor=executor)
