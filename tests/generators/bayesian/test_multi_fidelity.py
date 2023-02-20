from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from xopt.generators.bayesian.models.multi_fidelity import create_multifidelity_model
from xopt.generators.bayesian.multi_fidelity import MultiFidelityBayesianGenerator
from xopt.resources.testing import TEST_VOCS_BASE


class TestMultiFidelityGenerator:
    def test_init(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.constraints = []

        f_key = "s"

        options = MultiFidelityBayesianGenerator.default_options()
        options.model.fidelity_key = f_key

        MultiFidelityBayesianGenerator(vocs, options)

    def test_model_creation(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.constraints = []

        f_key = "s"
        data = pd.DataFrame(
            {
                "x1": np.random.randn(10),
                "x2": np.random.randn(10),
                "y1": np.random.randn(10),
                f_key: np.random.rand(10),
            }
        )

        create_multifidelity_model(data, vocs, f_key)

        # test bad fidelity values
        data = pd.DataFrame(
            {
                "x1": np.random.randn(10),
                "x2": np.random.randn(10),
                "y1": np.random.randn(10),
                f_key: np.random.rand(10) + 1.0,
            }
        )
        with pytest.raises(ValueError):
            create_multifidelity_model(data, vocs, f_key)

    def test_acq(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.constraints = []

        f_key = "s"

        options = MultiFidelityBayesianGenerator.default_options()
        options.model.fidelity_key = f_key

        generator = MultiFidelityBayesianGenerator(vocs, options)
        data = pd.DataFrame(
            {
                "x1": np.random.randn(10),
                "x2": np.random.randn(10),
                "y1": np.random.randn(10),
                f_key: np.random.rand(10),
            }
        )
        generator.add_data(data)

        acq = generator.get_acquisition(generator.model)

        # evaluate acquisition function at a test point
        batch_size = 1
        n_fantasies = 128
        test_x = torch.rand(3, batch_size + n_fantasies, 3).double()
        acq(test_x)

        # test total cost
        assert (
            generator.calculate_total_cost()
            == data[f_key].to_numpy().sum() + 10 * generator.options.acq.base_cost
        )

    def test_generation(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.constraints = []

        f_key = "s"

        options = MultiFidelityBayesianGenerator.default_options()
        options.model.fidelity_key = f_key
        options.optim.num_restarts = 1
        options.optim.raw_samples = 1
        options.acq.n_fantasies = 2

        generator = MultiFidelityBayesianGenerator(vocs, options)
        data = pd.DataFrame(
            {
                "x1": np.random.randn(2),
                "x2": np.random.randn(2),
                "y1": np.random.randn(2),
                f_key: np.random.rand(2),
            }
        )
        generator.add_data(data)

        # test single and and batch generation
        generator.generate(1)
        generator.generate(2)
