from copy import deepcopy

import pandas as pd
import pytest
import torch
from botorch.acquisition import ExpectedImprovement

from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.bayesian.expected_improvement import ExpectedImprovementGenerator
from xopt.generators.bayesian.objectives import CustomXoptObjective
from xopt.resources.testing import (
    TEST_VOCS_BASE,
    TEST_VOCS_DATA,
    check_generator_tensor_locations,
    create_set_options_helper,
    generate_without_warnings,
    xtest_callable,
)
from xopt.vocs import ObjectiveEnum, VOCS

cuda_combinations = [False] if not torch.cuda.is_available() else [False, True]
device_map = {False: torch.device("cpu"), True: torch.device("cuda:0")}


set_options = create_set_options_helper(data=TEST_VOCS_DATA)


class TestExpectedImprovement:
    def test_init(self):
        ei_gen = ExpectedImprovementGenerator(vocs=TEST_VOCS_BASE)
        ei_gen.model_dump_json()

    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_generate(self, use_cuda):
        gen = ExpectedImprovementGenerator(
            vocs=TEST_VOCS_BASE,
        )
        set_options(gen, use_cuda, add_data=True)

        candidate = generate_without_warnings(gen, 1)
        assert len(candidate) == 1

        candidate = generate_without_warnings(gen, 2)
        assert len(candidate) == 2

        check_generator_tensor_locations(gen, device_map[use_cuda])

    def test_generate_w_overlapping_objectives_constraints(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.constraints = {"y1": ["GREATER_THAN", 0.0]}
        test_vocs.observables = ["y1"]
        gen = ExpectedImprovementGenerator(
            vocs=test_vocs,
        )
        set_options(gen, add_data=True)

        candidate = generate_without_warnings(gen, 1)
        assert len(candidate) == 1

        candidate = generate_without_warnings(gen, 2)
        assert len(candidate) == 2

    def test_in_xopt(self):
        evaluator = Evaluator(function=xtest_callable)
        gen = ExpectedImprovementGenerator(
            vocs=TEST_VOCS_BASE,
        )
        set_options(gen)

        xopt = Xopt(generator=gen, evaluator=evaluator, vocs=TEST_VOCS_BASE)

        # initialize with single initial candidate
        xopt.random_evaluate(3)

        # now use bayes opt
        for _ in range(3):
            xopt.step()

    @pytest.mark.parametrize("use_cuda", cuda_combinations)
    def test_custom_objectives(self, use_cuda):
        train_x = torch.tensor([0.01, 0.3, 0.6, 0.99]).double()
        train_y = torch.sin(2 * torch.pi * train_x)
        train_c = torch.cos(2 * torch.pi * train_x)
        train_data = pd.DataFrame(
            {"x1": train_x.numpy(), "y1": train_y.numpy(), "c1": train_c.numpy()}
        )
        vocs = VOCS(**{"variables": {"x1": [0.0, 1.0]}, "observables": ["y1", "c1"]})

        class MyObjective(CustomXoptObjective):
            def forward(self, samples, X=None):
                return samples[..., self.vocs.output_names.index("y1")] ** 2

        gen = ExpectedImprovementGenerator(
            vocs=vocs, custom_objective=MyObjective(vocs)
        )
        set_options(gen, use_cuda)
        gen.add_data(train_data)
        best_f = gen._get_best_f(gen.data, gen.custom_objective)
        assert float(best_f) == float(torch.max(train_y**2))

        gen.generate(1)

        check_generator_tensor_locations(gen, device_map[use_cuda])

    def test_acquisition_accuracy(self):
        train_x = torch.tensor([0.01, 0.3, 0.6, 0.99]).double()
        train_y = torch.sin(2 * torch.pi * train_x)
        train_c = torch.cos(2 * torch.pi * train_x)
        train_data = pd.DataFrame(
            {"x1": train_x.numpy(), "y1": train_y.numpy(), "c1": train_c.numpy()}
        )
        test_x = torch.linspace(0.0, 1.0, 1000)

        for objective in ObjectiveEnum:
            vocs = VOCS(
                **{"variables": {"x1": [0.0, 1.0]}, "objectives": {"y1": objective}}
            )
            gen = ExpectedImprovementGenerator(vocs=vocs)
            set_options(gen)
            gen.add_data(train_data)
            model = gen.train_model().models[0]

            # xopt acquisition function - this is currently LogEI
            acq = gen.get_acquisition(model)

            # analytical EI acquisition function for no constraints
            # note that this cannot handle constraints or custom objectives
            assert acq.__class__.__name__ == "LogExpectedImprovement"

            # analytical acquisition function
            if objective == "MAXIMIZE":
                an_acq = ExpectedImprovement(model, best_f=train_y.max(), maximize=True)
            else:
                an_acq = ExpectedImprovement(
                    model, best_f=train_y.min(), maximize=False
                )

            # compare candidates (maximum in test data)
            with torch.no_grad():
                acq_v = acq(test_x.reshape(-1, 1, 1))
                candidate = test_x[torch.argmax(acq_v)]
                an_acq_v = an_acq(test_x.reshape(-1, 1, 1))
                an_candidate = test_x[torch.argmax(an_acq_v)]

            # difference should be small
            assert torch.abs(an_candidate - candidate) < 1e-6

        # test with constraints
        vocs = VOCS(
            **{
                "variables": {"x1": [0.0, 1.0]},
                "objectives": {"y1": "MAXIMIZE"},
                "constraints": {"c1": ["GREATER_THAN", 100.0]},
            }
        )
        gen = ExpectedImprovementGenerator(vocs=vocs)
        set_options(gen)
        gen.add_data(train_data)

        model = gen.train_model()

        # acquisition function computed by the EI generator should be qLogExpectedImprovement
        acq = gen.get_acquisition(model)
        assert acq.__class__.__name__ == "qLogExpectedImprovement"
        assert acq._constraints is not None

        # exp of acquisition values should be near zero for all points since constraint is
        # impossible to satisfy
        exp_acq_values = acq(test_x.reshape(-1, 1, 1)).exp()
        assert torch.all(exp_acq_values > 0.0)
        assert torch.all(exp_acq_values < 1e-6)

        # acquisition function should be nearly identical to unconstrained 
        # case if the constraint is always satisfied
        vocs_unconstrained = VOCS(
            **{
                "variables": {"x1": [0.0, 1.0]},
                "objectives": {"y1": "MAXIMIZE"},
            }
        )
        vocs_always_satisfied = VOCS(
            **{
                "variables": {"x1": [0.0, 1.0]},
                "objectives": {"y1": "MAXIMIZE"},
                "constraints": {"c1": ["LESS_THAN", 100]},
            }
        )
        acq_values = []
        for v in [vocs_unconstrained, vocs_always_satisfied]:
            gen = ExpectedImprovementGenerator(vocs=v)
            set_options(gen)
            gen.add_data(train_data)
            gen.n_monte_carlo_samples = 512

            model = gen.train_model()

            acq = gen.get_acquisition(model)
            acq_values.append(acq(test_x.reshape(-1, 1, 1)).exp())

        assert torch.allclose(acq_values[0].double(), acq_values[1].double(), atol=1e-3)
