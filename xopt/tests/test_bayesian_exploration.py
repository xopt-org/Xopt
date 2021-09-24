import copy

from xopt import Xopt
from .evaluators import TNK


class TestClassBayesExp:
    VOCS = TNK.VOCS
    config = {'vocs': TNK.VOCS.copy()}
    config['simulation'] = {'name': 'test_TNK',
                            'evaluate': 'xopt.tests.evaluators.TNK.evaluate_TNK'}
    config['xopt'] = {'output_path': '', 'verbose': False}
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

    def test_batch(self):
        test_config = copy.deepcopy(self.config)
        test_config['algorithm']['options']['generator_options'] = {
            'batch_size': 2
        }

        X = Xopt(test_config)
        X.run()
