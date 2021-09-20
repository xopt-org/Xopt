import copy

from xopt import Xopt
import pytest
from .evaluators import TNK


class TestClassMOBO:
    VOCS = TNK.VOCS
    config = {'vocs': TNK.VOCS.copy()}
    config['simulation'] = {'name': 'test_TNK',
                            'evaluate': 'xopt.tests.evaluators.TNK.evaluate_TNK'}
    config['xopt'] = {'output_path': '', 'verbose': False}
    config['algorithm'] = {'name': 'mobo',
                           'options': {
                               'n_initial_samples': 1,
                               'n_steps': 1,
                               'ref': [],
                               'generator_options': {
                                   'batch_size': 1,
                                   'mc_samples': 4,
                                   'num_restarts': 1,
                                   'raw_samples': 4
                               }
                           }}

    def test_mobo_base(self):
        test_config = copy.deepcopy(self.config)

        # try without reference point
        with pytest.raises(ValueError):
            X = Xopt(test_config)
            X.run()

        # try with reference point
        test_config['algorithm']['options']['ref'] = [1.4, 1.4]
        X = Xopt(test_config)
        X.run()

        # try with bad reference point
        with pytest.raises(ValueError):
            test_config['algorithm']['options']['ref'] = [1.4, 1.4, 1.4]
            X = Xopt(test_config)
            X.run()

    def test_mobo_batch(self):
        test_config = copy.deepcopy(self.config)
        test_config['algorithm']['options']['generator_options'] = {
            'batch_size': 2
        }

        # try with reference point
        test_config['algorithm']['options']['ref'] = [1.4, 1.4]
        X = Xopt(test_config)
        X.run()

    def test_mobo_proximal(self):
        test_config = copy.deepcopy(self.config)
        test_config['algorithm']['options']['generator_options'] = {
            'sigma': [1.0, 1.0],
            'batch_size': 2
        }
        test_config['algorithm']['options']['ref'] = [1.4, 1.4]

        # try with sigma matrix
        X = Xopt(test_config)
        X.run()

    def test_mobo_unconstrained(self):
        test_config = copy.deepcopy(self.config)
        test_config['vocs']['constraints'] = {}
        test_config['algorithm']['options']['ref'] = [1.4, 1.4]

        # try with sigma matrix
        X = Xopt(test_config)
        X.run()
