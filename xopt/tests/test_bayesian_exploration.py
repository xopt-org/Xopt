import copy

import pytest

from xopt import Xopt
from .evaluators import TNK
from ..bayesian.utils import UnsupportedError

class TestClassBayesExp:
    VOCS = TNK.VOCS
    config = {'vocs': TNK.VOCS.copy()}
    config['simulation'] = {'name': 'test_TNK',
                            'evaluate': 'xopt.tests.evaluators.TNK.evaluate_TNK'}
    config['xopt'] = {'output_path': ''}
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

        #run with proximal term
        X.algorithm['options']['generator_options']['sigma'] = [1, 1]
        X.run()

        #run with bad proximal term
        X.algorithm['options']['generator_options']['sigma'] = [1, 1, 1]
        with pytest.raises(ValueError):
            X.run()

    def test_batch(self):
        test_config = copy.deepcopy(self.config)
        test_config['algorithm']['options']['generator_options'] = {
            'batch_size': 2
        }

        X = Xopt(test_config)
        X.run()

        # try to add proximal term
        X.algorithm['options']['generator_options']['sigma'] = [1, 1]
        with pytest.raises(UnsupportedError):
            X.run()

    def test_initial_x(self):
        test_config = copy.deepcopy(self.config)
        X = Xopt(test_config)
        X.algorithm['options']['initial_x'] = [1, 1]
        X.run()
