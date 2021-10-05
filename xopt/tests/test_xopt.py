from copy import deepcopy

from ..xopt import Xopt
import pytest


class TestXoptConfig:
    def test_xopt_default(self):
        X = Xopt()

        # check default configs
        assert list(X.config.keys()) == ['xopt',
                                         'algorithm',
                                         'simulation',
                                         'vocs']

        # check bad config files
        bad_config = {'xopt': None,
                      'algorithm': None}
        with pytest.raises(Exception):
            X = Xopt(bad_config)

    def test_bad_configs(self):
        X = Xopt()
        default_config = X.config

        # test allowable keys
        for name in default_config.keys():
            with pytest.raises(Exception):
                new_config = deepcopy(default_config)
                new_config[name].update({'random_key': None})
                X2 = Xopt(new_config)

    def test_algorithm_config(self):
        # test algorithm specification
        X = Xopt()
        with pytest.raises(ValueError):
            X.configure_algorithm()

        # retry with a bad function name
        with pytest.raises(Exception):
            X.algorithm['function'] = 'dummy'
            X.configure_algorithm()

        # retry with bad module
        X.algorithm['function'] = 'dummy.dummy'
        with pytest.raises(ModuleNotFoundError):
            X.configure_algorithm()

    def test_simulation_config(self):
        X = Xopt()

        # default has no function
        with pytest.raises(ValueError):
            X.configure_simulation()

        # specify a good function
        X.simulation['evaluate'] = lambda x: x
        X.configure_simulation()

        # specify a bad function
        X.simulation['evaluate'] = lambda x, y: x
        with pytest.raises(AssertionError):
            X.configure_simulation()

        def dummy(x, y=None):
            return x
        X.simulation['evaluate'] = dummy
        X.configure_simulation()
        assert X.config['simulation']['options'] == {'y': None}

    def test_vocs_config(self):
        X = Xopt()
        X.configure_vocs()






