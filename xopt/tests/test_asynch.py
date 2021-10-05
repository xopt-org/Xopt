from xopt import Xopt
from .evaluators import quad_3d
from concurrent.futures import ThreadPoolExecutor
import copy

class TestAsynchMultiFidelity:
    VOCS = quad_3d.VOCS
    config = {'vocs': quad_3d.VOCS.copy()}
    config['simulation'] = {'name': 'Quad 3D',
                            'evaluate': 'xopt.tests.evaluators.quad_3d.evaluate'}
    config['xopt'] = {'output_path': ''}
    config['algorithm'] = {'name': 'multi_fidelity',
                           'options': {
                               'processes': 2,
                               'budget': 3,
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
        test_config['algorithm']['options']['restart_file'] = \
            'xopt/tests/asynch_test_results.json'
        X = Xopt(test_config)
        executor = ThreadPoolExecutor()
        X.run(executor=executor)
