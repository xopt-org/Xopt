import time
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import torch.cuda

from xopt.generators.bayesian.time_dependent import TimeDependentBayesianGenerator
from xopt.generators.bayesian.upper_confidence_bound import (
    TDUpperConfidenceBoundGenerator,
)
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA


class TestTimeDependentBO:
    @patch.multiple(TimeDependentBayesianGenerator, __abstractmethods__=set())
    def test_init(self):
        TimeDependentBayesianGenerator(vocs=TEST_VOCS_BASE)

    @patch.multiple(TimeDependentBayesianGenerator, __abstractmethods__=set())
    def test_model_generation(self):
        gen = TimeDependentBayesianGenerator(vocs=TEST_VOCS_BASE)
        test_data = deepcopy(TEST_VOCS_DATA)

        time_array = []
        for i in range(len(test_data)):
            time_array.append(time.time())
            time.sleep(0.01)

        test_data["time"] = np.array(time_array)

        model = gen.train_model(test_data)

        # make sure time data is in the last model
        assert np.all(
            model.models[-1]
            .input_transform.untransform(model.models[-1].train_inputs[0])[:, -1]
            .numpy()
            == test_data["time"].to_numpy().flatten()
        )

    def test_td_ucb(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)

        gen = TDUpperConfidenceBoundGenerator(vocs=test_vocs)
        gen.added_time = 5.0
        gen.n_monte_carlo_samples = 1

        test_data = deepcopy(TEST_VOCS_DATA)
        time_array = []
        for i in range(len(test_data)):
            time_array.append(time.time())
            time.sleep(0.01)

        test_data["time"] = np.array(time_array)
        gen.add_data(test_data)
        gen.generate(1)

        # test without constraints
        test_vocs.constraints = {}
        gen.added_time = 5.0
        gen.n_monte_carlo_samples = 1

        test_data = deepcopy(TEST_VOCS_DATA)
        time_array = []
        for i in range(len(test_data)):
            time_array.append(time.time())
            time.sleep(0.01)

        test_data["time"] = np.array(time_array)

        gen.add_data(test_data)
        gen.generate(1)

    def test_cuda_td_ucb(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)

        gen = TDUpperConfidenceBoundGenerator(vocs=test_vocs)
        if torch.cuda.is_available():
            gen.use_cuda = True
            gen.added_time = 5.0
            gen.n_monte_carlo_samples = 1

            test_data = deepcopy(TEST_VOCS_DATA)
            time_array = []
            for i in range(len(test_data)):
                time_array.append(time.time())
                time.sleep(0.01)

            test_data["time"] = np.array(time_array)

            gen.add_data(test_data)
            gen.generate(1)
