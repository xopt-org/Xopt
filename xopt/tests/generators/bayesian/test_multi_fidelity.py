from copy import deepcopy

import pytest
import torch
from pandas import Series

from xopt.generators.bayesian.multi_fidelity import MultiFidelityGenerator
from xopt.resources.test_functions.tnk import tnk_vocs
from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA


class TestMultiFidelityGenerator:
    def test_init(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.constraints = {}

        for ele in ["MAXIMIZE", "MINIMIZE"]:
            vocs.objectives = {"y1": ele}
            gen = MultiFidelityGenerator(vocs=vocs)

            # test reference point
            pt = gen.reference_point
            assert gen.vocs.objective_names == ["y1", "s"]

            if ele == "MAXIMIZE":
                assert pt == {"s": 0.0, "y1": -100.0}
            else:
                assert pt == {"s": 0.0, "y1": 100.0}

            # test fidelity objective index
            assert gen.fidelity_objective_index == 1
            assert gen.fidelity_variable_index == 2

        vocs.objectives = {"y1": "EXPLORE"}
        with pytest.raises(ValueError):
            MultiFidelityGenerator(vocs=vocs)

    def test_add_data(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.constraints = {}

        gen = MultiFidelityGenerator(vocs=vocs)

        # add a fidelity parameter to data
        data = deepcopy(TEST_VOCS_DATA)
        fidelity_parameter = "s"
        data[fidelity_parameter] = Series([1.0] * 10)

        gen.add_data(data)

        # try with bad data
        bad_data = deepcopy(data)
        bad_data[fidelity_parameter] = 10.0 * bad_data[fidelity_parameter]
        with pytest.raises(ValueError):
            gen.add_data(bad_data)

        # try with data missing fidelity parameter
        with pytest.raises(ValueError):
            gen.add_data(TEST_VOCS_DATA)

    def test_model_creation(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.constraints = {}

        data = deepcopy(TEST_VOCS_DATA)

        # add a fidelity parameter
        fidelity_parameter = "s"
        data[fidelity_parameter] = Series([1.0] * 10)

        generator = MultiFidelityGenerator(
            vocs=vocs, fidelity_parameter=fidelity_parameter
        )
        generator.add_data(data)

        generator.train_model(generator.data)

    def test_acq(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.constraints = {}

        data = deepcopy(TEST_VOCS_DATA)

        # add a fidelity parameter
        fidelity_parameter = "s"
        data[fidelity_parameter] = Series([1.0] * 10)

        generator = MultiFidelityGenerator(vocs=vocs)
        generator.add_data(data)
        generator.train_model()

        acq = generator.get_acquisition(generator.model)

        # evaluate acquisition function at a test point
        test_x = torch.rand(3, 1, 3).double()
        acq(test_x)

        # test total cost
        assert (
            generator.calculate_total_cost()
            == data[fidelity_parameter].to_numpy().sum() + 10
        )

        # test model validation
        with pytest.raises(ValueError):
            generator.get_acquisition(None)

    def test_generation(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.constraints = {}

        data = deepcopy(TEST_VOCS_DATA)

        for ele in ["MAXIMIZE", "MINIMIZE"]:
            vocs.objectives = {"y1": ele}

            # add a fidelity parameter
            fidelity_parameter = "s"
            data[fidelity_parameter] = Series([1.0] * 10)

            generator = MultiFidelityGenerator(vocs=vocs)
            generator.numerical_optimizer.n_restarts = 1

            generator.add_data(data)

            # test getting the objective
            generator._get_objective()

            # test single and and batch generation
            generator.generate(1)
            generator.generate(2)

            # test getting the best point at max fidelity
            # get optimal value at max fidelity
            generator.get_optimum()

    def test_multi_objective(self):
        my_vocs = deepcopy(tnk_vocs)
        my_vocs.constraints = {}
        MultiFidelityGenerator(vocs=my_vocs, reference_point={"y1": 1.5, "y2": 1.5})
