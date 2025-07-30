import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)

from xopt.base import Xopt
from xopt.errors import XoptError
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.mobo import MOBOGenerator
from xopt.numerical_optimizer import GridOptimizer
from xopt.resources.test_functions.tnk import (
    evaluate_TNK,
    tnk_reference_point,
    tnk_vocs,
)
from xopt.resources.testing import (
    TEST_VOCS_BASE_MO,
    TEST_VOCS_BASE_MO_NC,
    TEST_VOCS_DATA_MO,
    TEST_VOCS_REF_POINT,
    check_dict_allclose,
    check_generator_tensor_locations,
    check_dict_equal,
    create_set_options_helper,
    reload_gen_from_json,
    reload_gen_from_yaml,
)

cuda_combinations = [False] if not torch.cuda.is_available() else [False, True]
device_map = {False: torch.device("cpu"), True: torch.device("cuda:0")}

set_options = create_set_options_helper(data=TEST_VOCS_DATA_MO)


class TestMOBOGenerator:
    def test_init(self):
        MOBOGenerator(vocs=TEST_VOCS_BASE_MO, reference_point=TEST_VOCS_REF_POINT)

        # test bad reference point
        with pytest.raises(XoptError):
            MOBOGenerator(vocs=TEST_VOCS_BASE_MO, reference_point={})

    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_generate(self, use_cuda):
        gen = MOBOGenerator(vocs=TEST_VOCS_BASE_MO, reference_point=TEST_VOCS_REF_POINT)
        set_options(gen, use_cuda, add_data=True)

        candidate = gen.generate(1)
        assert len(candidate) == 1

        candidate = gen.generate(2)
        assert len(candidate) == 2

        check_generator_tensor_locations(gen, device_map[use_cuda])

        gen = MOBOGenerator(
            vocs=TEST_VOCS_BASE_MO_NC, reference_point=TEST_VOCS_REF_POINT
        )
        set_options(gen, use_cuda, add_data=True)

        candidate = gen.generate(1)
        assert len(candidate) == 1

        candidate = gen.generate(2)
        assert len(candidate) == 2

        check_generator_tensor_locations(gen, device_map[use_cuda])

    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_round_trip(self, use_cuda):
        gen = MOBOGenerator(vocs=TEST_VOCS_BASE_MO, reference_point=TEST_VOCS_REF_POINT)
        set_options(gen, use_cuda)
        gen.add_data(TEST_VOCS_DATA_MO)
        gen.generate(1)

        gen2 = reload_gen_from_json(gen)
        gen3 = reload_gen_from_yaml(gen)

        torch.manual_seed(42)
        candidate1_2 = gen.generate(1)
        torch.manual_seed(42)
        candidate2_2 = gen2.generate(1)
        torch.manual_seed(42)
        candidate3_2 = gen3.generate(1)

        check_dict_equal(
            json.loads(gen.json()),
            json.loads(gen2.json()),
            excluded_keys=["computation_time", "pareto_front_history"],
        )
        check_dict_equal(
            json.loads(gen.json()),
            json.loads(gen3.json()),
            excluded_keys=["computation_time", "pareto_front_history"],
        )

        # this fails almost always without manual seed!
        check_dict_allclose(candidate1_2[0], candidate2_2[0], rtol=0)
        check_dict_allclose(candidate1_2[0], candidate3_2[0], rtol=0)

        check_generator_tensor_locations(gen, device_map[use_cuda])
        check_generator_tensor_locations(gen2, device_map[use_cuda])
        check_generator_tensor_locations(gen3, device_map[use_cuda])

    def test_script(self):
        evaluator = Evaluator(function=evaluate_TNK)
        reference_point = tnk_reference_point

        gen = MOBOGenerator(
            vocs=tnk_vocs,
            reference_point=reference_point,
            numerical_optimizer=GridOptimizer(n_grid_points=2),
        )
        gen = deepcopy(gen)
        gen.n_monte_carlo_samples = 1

        for ele in [gen]:
            dump = ele.model_dump()
            generator = MOBOGenerator(vocs=tnk_vocs, **dump)
            X = Xopt(generator=generator, evaluator=evaluator, vocs=tnk_vocs)
            X.random_evaluate(3)
            X.step()

    def test_pareto_front_calculation(self):
        test_data = pd.DataFrame(
            {
                "x1": [0.1, 0.2, 0.4, 0.4],
                "x2": [0.1, 0.2, 0.3, 0.2],
                "y1": [1.0, 2.0, 1.0, 0.0],
                "y2": [0.5, 0.1, 1.0, 1.5],
            }
        )
        reference_point = {"y1": 10.0, "y2": 1.5}
        gen = MOBOGenerator(
            vocs=TEST_VOCS_BASE_MO_NC,
            reference_point=reference_point,
            use_pf_as_initial_points=True,
        )
        gen.add_data(test_data)

        pfx, pfy, _, _ = gen.get_pareto_front_and_hypervolume()
        assert torch.allclose(
            torch.tensor([[0.1, 0.2, 0.4], [0.1, 0.2, 0.2]], dtype=torch.double).T, pfx
        )
        assert torch.allclose(
            torch.tensor([[1.0, 2.0, 0.0], [0.5, 0.1, 1.5]], dtype=torch.double).T, pfy
        )

        # test pf history tracking
        gen.update_pareto_front_history()
        assert len(gen.pareto_front_history) == 4
        assert gen.pareto_front_history["n_non_dominated"].to_list() == [1, 2, 2, 3]

        # make sure that the pareto front is not updated if there are no new points
        gen.update_pareto_front_history()
        assert len(gen.pareto_front_history) == 4
        assert gen.pareto_front_history["n_non_dominated"].to_list() == [1, 2, 2, 3]

        # test where all the points are dominated by the reference point
        test_data = pd.DataFrame(
            {
                "x1": [0.1, 0.2, 0.4, 0.4],
                "x2": [0.1, 0.2, 0.3, 0.2],
                "y1": [100.0, 2.0, 100.0, 10.0],
                "y2": [0.5, 2.0, 2.0, 1.5],
            }
        )
        gen = MOBOGenerator(
            vocs=TEST_VOCS_BASE_MO_NC,
            reference_point=reference_point,
            use_pf_as_initial_points=True,
        )
        gen.add_data(test_data)

        pfx, pfy, _, _ = gen.get_pareto_front_and_hypervolume()
        assert pfx is None
        assert pfy is None

        # test updating the historical pareto front
        gen.update_pareto_front_history()
        assert len(gen.pareto_front_history) == 4
        assert gen.pareto_front_history["n_non_dominated"].to_list() == [0, 0, 0, 0]
        assert gen.pareto_front_history["hypervolume"].to_list() == [0.0, 0.0, 0.0, 0.0]

        # test with constraints
        test_data = pd.DataFrame(
            {
                "x1": [0.1, 0.2, 0.4, 0.4],
                "x2": [0.1, 0.2, 0.3, 0.2],
                "y1": [1.0, 2.0, 1.0, 0.0],
                "y2": [0.5, 0.1, 1.0, 1.5],
                "c1": [1.0, 1.0, 1.0, 0.0],
            }
        )
        gen = MOBOGenerator(
            vocs=TEST_VOCS_BASE_MO,
            reference_point=reference_point,
            use_pf_as_initial_points=True,
        )
        gen.add_data(test_data)
        pfx, pfy, _, _ = gen.get_pareto_front_and_hypervolume()
        assert torch.allclose(
            torch.tensor([[0.1, 0.2], [0.1, 0.2]], dtype=torch.double).T, pfx
        )
        assert torch.allclose(
            torch.tensor([[1.0, 2.0], [0.5, 0.1]], dtype=torch.double).T, pfy
        )

        # test pf history tracking
        gen.update_pareto_front_history()
        assert len(gen.pareto_front_history) == 4
        assert gen.pareto_front_history["n_non_dominated"].to_list() == [1, 2, 2, 2]

    def test_hypervolume_calculation(self):
        vocs = deepcopy(TEST_VOCS_BASE_MO_NC)

        data = pd.DataFrame(
            {
                "x1": np.random.rand(2),
                "x2": np.random.rand(2),
                "y1": np.array((1.0, 0.0)),
                "y2": np.array((0.0, 2.0)),
            }
        )
        reference_point = {"y1": 10.0, "y2": 1.0}
        gen = MOBOGenerator(vocs=vocs, reference_point=reference_point)
        gen.add_data(data)

        assert gen.get_pareto_front_and_hypervolume()[-1] == 9.0

        vocs.objectives["y1"] = "MAXIMIZE"
        gen = MOBOGenerator(vocs=vocs, reference_point=reference_point)
        gen.add_data(data)

        assert gen.get_pareto_front_and_hypervolume()[-1] == 0.0

    def test_initial_conditions(self):
        test_data = pd.DataFrame(
            {
                "x1": [0.1, 0.2, 0.4, 0.4],
                "x2": [0.1, 0.2, 0.3, 0.2],
                "y1": [1.0, 2.0, 1.0, 0.0],
                "y2": [0.5, 0.1, 1.0, 1.5],
            }
        )
        reference_point = {"y1": 10.0, "y2": 1.5}
        gen = MOBOGenerator(
            vocs=TEST_VOCS_BASE_MO_NC,
            reference_point=reference_point,
            use_pf_as_initial_points=True,
        )
        gen.n_monte_carlo_samples = 1
        gen.add_data(test_data)
        initial_points = gen._get_initial_conditions()

        assert torch.allclose(
            torch.tensor([[0.1, 0.2, 0.4], [0.1, 0.2, 0.2]], dtype=torch.double).T,
            initial_points[:3].squeeze(),
        )
        assert len(initial_points) == gen.numerical_optimizer.n_restarts
        gen.generate(1)

        # try with a small number of n_restarts
        gen.numerical_optimizer.n_restarts = 1
        initial_points = gen._get_initial_conditions()
        assert len(initial_points) == 1
        gen.generate(1)

        # try with no points on the pareto front
        gen.reference_point = {"y1": 0.0, "y2": 0.0}
        gen.numerical_optimizer.n_restarts = 20

        initial_points = gen._get_initial_conditions()
        assert initial_points is None
        gen.generate(1)

        # test with constraints
        test_data = pd.DataFrame(
            {
                "x1": [0.1, 0.2, 0.4, 0.4, 0.15],
                "x2": [0.1, 0.2, 0.3, 0.2, 0.15],
                "y1": [1.0, 2.0, 1.0, 0.0, 1.5],
                "y2": [0.5, 0.1, 1.0, 1.5, 0.25],
                "c1": [1.0, 1.0, 1.0, 1.0, 0.0],
            }
        )
        gen = MOBOGenerator(
            vocs=TEST_VOCS_BASE_MO,
            reference_point=reference_point,
            use_pf_as_initial_points=True,
        )
        gen.add_data(test_data)
        gen.numerical_optimizer.max_time = 1.0

        # make sure that no invalid points make it into the initial conditions
        ic = gen._get_initial_conditions()
        assert not torch.allclose(
            ic[:4],
            torch.tensor(((0.1, 0.1), (0.2, 0.2), (0.4, 0.2), (0.15, 0.15)))
            .reshape(4, 1, 2)
            .double(),
        )

        gen.generate(1)

    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_log_mobo(self, use_cuda):
        evaluator = Evaluator(function=evaluate_TNK)
        reference_point = tnk_reference_point

        gen = MOBOGenerator(
            vocs=tnk_vocs,
            reference_point=reference_point,
        )
        gen.use_cuda = use_cuda
        gen = deepcopy(gen)
        gen.n_monte_carlo_samples = 20

        for ele in [gen]:
            dump = ele.model_dump()
            generator = MOBOGenerator(vocs=tnk_vocs, **dump)
            X = Xopt(generator=generator, evaluator=evaluator, vocs=tnk_vocs)
            X.generator.numerical_optimizer.max_iter = 1
            X.random_evaluate(3)
            X.step()

            assert isinstance(
                X.generator.get_acquisition(X.generator.model),
                qLogNoisyExpectedHypervolumeImprovement,
            )

        check_generator_tensor_locations(gen, device_map[use_cuda])

    def test_objective_constraint_nans(self):
        test_data = pd.DataFrame(
            {
                "x1": [0.1, 0.2, 0.4, 0.4, 0.15],
                "x2": [0.1, 0.2, 0.3, 0.2, 0.15],
                "y1": [1.0, 2.0, 1.0, 0.0, 1.5],
                "y2": [0.5, 0.1, np.nan, 1.5, 0.25],
                "c1": [1.0, 1.0, 1.0, np.nan, 0.0],
            }
        )

        reference_point = {"y1": 10.0, "y2": 1.5}
        gen = MOBOGenerator(
            vocs=TEST_VOCS_BASE_MO,
            reference_point=reference_point,
            n_monte_carlo_samples=1,
        )
        gen.numerical_optimizer.max_iter = 1
        gen.add_data(test_data)
        gen.generate(1)
