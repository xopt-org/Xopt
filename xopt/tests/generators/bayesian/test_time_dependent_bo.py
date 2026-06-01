import pytest
import time
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from xopt.generators.bayesian.time_dependent import TimeDependentBayesianGenerator
from xopt.generators.bayesian.models.time_dependent import TimeDependentModelConstructor
from xopt.generators.bayesian.upper_confidence_bound import (
    TDUpperConfidenceBoundGenerator,
)
from xopt.vocs import VOCS
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA


class TestTimeDependentBO:
    @patch.multiple(TimeDependentBayesianGenerator, __abstractmethods__=set())
    def test_init(self):
        TimeDependentBayesianGenerator(vocs=TEST_VOCS_BASE)

    @patch.multiple(TimeDependentBayesianGenerator, __abstractmethods__=set())
    def test_model_generation(self):
        # test single dim variable space
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.variables = {"x1": [0, 1.0]}
        gen = TimeDependentBayesianGenerator(vocs=vocs)
        test_data = deepcopy(TEST_VOCS_DATA)
        test_data.drop("x2", axis=1, inplace=True)

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

        # test multi-dim variable space
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
        gen.added_time = 0.1
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
        gen.added_time = 0.1
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

    def test_initialize_spectral_kernel_from_data(self):
        # Create a minimal VOCS object
        vocs = VOCS(variables={"x1": [0.0, 1.0]}, objectives={"y1": "MINIMIZE"})

        # Create a minimal dataset with only one data point
        data = pd.DataFrame({"x1": [0.5], "time": [1.0], "y1": [0.0]})

        model_constructor = TimeDependentModelConstructor(
            initialize_spectral_kernel_from_data=True
        )

        # Expect a RuntimeWarning due to insufficient training data
        with pytest.raises(
            RuntimeWarning,
            match="cannot initialize spectral kernel from a single data sample",
        ):
            model_constructor.build_model_from_vocs(
                vocs, data, dtype=torch.double, device="cpu"
            )

        # Create a dataset with more than one data point
        data = pd.DataFrame({"x1": [0.5, 0.6], "time": [1.0, 2.0], "y1": [0.0, -1.0]})

        # Initialize the model constructor with spectral kernel initialization enabled
        model_constructor = TimeDependentModelConstructor(
            initialize_spectral_kernel_from_data=True
        )
        model_constructor.build_model_from_vocs(
            vocs, data, dtype=torch.double, device="cpu"
        )

    @patch.multiple(TimeDependentBayesianGenerator, __abstractmethods__=set())
    def test_gp_constructor_assignment(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.variables = {"x1": [0, 1.0]}

        # Test with None
        generator = TimeDependentBayesianGenerator(vocs=vocs)
        generator.gp_constructor = None
        assert isinstance(generator.gp_constructor, TimeDependentModelConstructor)

        # Test with TimeDependentModelConstructor instance
        constructor = TimeDependentModelConstructor()
        generator.gp_constructor = constructor
        assert generator.gp_constructor is constructor

        # Test with valid string
        generator.gp_constructor = "time_dependent"
        assert isinstance(generator.gp_constructor, TimeDependentModelConstructor)

        # Test with valid dictionary
        generator.gp_constructor = {"name": "time_dependent"}
        assert isinstance(generator.gp_constructor, TimeDependentModelConstructor)

        # Test with invalid string
        with pytest.raises(ValueError, match="invalid_constructor not found"):
            generator.gp_constructor = "invalid_constructor"

        # Test with invalid dictionary
        with pytest.raises(
            ValueError, match="Constructor 'invalid_constructor' not found"
        ):
            generator.gp_constructor = {"name": "invalid_constructor"}
