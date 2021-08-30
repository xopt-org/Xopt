import copy

from xopt import Xopt
import pytest
from ..evaluators import test_TNK


class TestClassBayesExp:
    VOCS = test_TNK.VOCS
    config = {'vocs': test_TNK.VOCS.copy()}
    config['simulation'] = {'name': 'test_TNK',
                            'evaluate': 'xopt.evaluators.test_TNK.evaluate_TNK'}
    config['xopt'] = {'output_path': '', 'verbose': False, 'algorithm': 'mobo'}
    config['algorithm'] = {'name': 'bayesian_exploration',
                           'options': {
                               'n_initial_samples': 1,
                               'n_steps': 1,
                               'generator_options': {
                                   'mc_samples': 4,
                                   'num_restarts': 1,
                                   'raw_samples': 4
                               }
                           }}

    def test_bayes_exp_base(self):
        test_config = copy.deepcopy(self.config)
        X = Xopt(test_config)
        X.run()

        X = Xopt(test_config)
        X.run()

    def test_batch(self):
        test_config = copy.deepcopy(self.config)
        test_config['algorithm']['options']['generator_options'] = {
            'batch_size': 2
        }

        X = Xopt(test_config)
        X.run()


