import copy

from xopt import Xopt
from .evaluators import multi_fidelity


class TestClassMultiFidelity:
    VOCS = multi_fidelity.VOCS
    config = {'vocs': multi_fidelity.VOCS.copy()}
    config['simulation'] = {'name': 'AugmentedHartmann',
                            'evaluate':
                                'xopt.tests.evaluators.multi_fidelity.evaluate'}
    config['xopt'] = {'output_path': '', 'verbose': True}
    config['algorithm'] = {'name': 'multi_fidelity',
                           'options': {
                               'budget': 2,
                               'processes': 1,
                               'generator_options': {
                                   'mc_samples': 4,
                                   'num_restarts': 1,
                                   'raw_samples': 4
                               }
                           }}

    def test_multi_fidelity_base(self):
        test_config = copy.deepcopy(self.config)
        X = Xopt(test_config)
        X.run()

    def test_batch(self):
        test_config = copy.deepcopy(self.config)
        test_config['algorithm']['options']['processes'] = 2

        X = Xopt(test_config)
        X.run()
