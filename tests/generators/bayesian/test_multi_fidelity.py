from copy import deepcopy

import pytest
import torch
from pandas import Series

from xopt.generators.bayesian.multi_fidelity import MultiFidelityGenerator
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA


class TestMultiFidelityGenerator:
    def test_init(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.constraints = {}

        fidelity_parameter = "s"

        options = MultiFidelityGenerator.default_options()
        options.model.fidelity_parameter = fidelity_parameter

        gen = MultiFidelityGenerator(vocs, options)

        # test reference point
        pt = gen.reference_point
        assert torch.allclose(pt, torch.tensor((0.0, -10.0)).to(pt))

    def test_model_creation(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.constraints = {}

        data = deepcopy(TEST_VOCS_DATA)

        # add a fidelity parameter
        fidelity_parameter = "s"
        data[fidelity_parameter] = Series([1.0] * 10)

        options = MultiFidelityGenerator.default_options()
        options.model.fidelity_parameter = fidelity_parameter

        generator = MultiFidelityGenerator(vocs, options)
        generator.add_data(data)

        generator.train_model(generator.data)

        # try to add bad data
        bad_data = deepcopy(TEST_VOCS_DATA)
        bad_data[fidelity_parameter] = Series([10.0] * 10)
        generator = MultiFidelityGenerator(vocs, options)
        with pytest.raises(ValueError):
            generator.add_data(bad_data)

        bad_data = deepcopy(TEST_VOCS_DATA)
        bad_data[fidelity_parameter] = Series([-1.0] * 10)
        generator = MultiFidelityGenerator(vocs, options)
        with pytest.raises(ValueError):
            generator.add_data(bad_data)

    def test_acq(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.constraints = {}

        data = deepcopy(TEST_VOCS_DATA)

        # add a fidelity parameter
        fidelity_parameter = "s"
        data[fidelity_parameter] = Series([1.0] * 10)

        options = MultiFidelityGenerator.default_options()
        options.model.fidelity_parameter = fidelity_parameter

        generator = MultiFidelityGenerator(vocs, options)
        generator.add_data(data)

        acq = generator.get_acquisition(generator.model)

        # evaluate acquisition function at a test point
        test_x = torch.rand(3, 1, 3).double()
        acq(test_x)

        # test total cost
        assert (
            generator.calculate_total_cost()
            == data[fidelity_parameter].to_numpy().sum() + 10
        )

    def test_generation(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.constraints = {}

        data = deepcopy(TEST_VOCS_DATA)

        # add a fidelity parameter
        fidelity_parameter = "s"
        data[fidelity_parameter] = Series([1.0] * 10)

        options = MultiFidelityGenerator.default_options()
        options.model.fidelity_parameter = fidelity_parameter
        options.optim.num_restarts = 1
        options.optim.raw_samples = 1

        generator = MultiFidelityGenerator(vocs, options)
        generator.add_data(data)

        # test getting the objective
        generator._get_objective()

        # test single and and batch generation
        generator.generate(1)
        generator.generate(2)

        # test getting the best point at max fidelity
        # get optimal value at max fidelity
        generator.get_optimum()
