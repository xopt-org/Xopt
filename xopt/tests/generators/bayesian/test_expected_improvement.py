from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition import FixedFeatureAcquisitionFunction

from gest_api.vocs import (
    VOCS,
)

from xopt.base import Xopt
from xopt.errors import VOCSError
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

        xopt = Xopt(generator=gen, evaluator=evaluator)

        # initialize with initial candidates
        np.random.seed(42)
        xopt.random_evaluate(10)

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

        # should raise an error if no custom objective is provided
        with pytest.raises(VOCSError):
            ExpectedImprovementGenerator(vocs=vocs)

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

        for objective in ["MINIMIZE", "MAXIMIZE"]:
            vocs = VOCS(
                **{"variables": {"x1": [0.0, 1.0]}, "objectives": {"y1": objective}}
            )
            gen = ExpectedImprovementGenerator(vocs=vocs)
            set_options(gen)
            gen.n_monte_carlo_samples = 512
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

            # compare acquisition values
            with torch.no_grad():
                acq_v = acq(test_x.reshape(-1, 1, 1)).exp()
                an_acq_v = an_acq(test_x.reshape(-1, 1, 1))

            # difference should be small
            assert torch.allclose(acq_v.double(), an_acq_v.double(), atol=1e-4)

        # test with constraints
        vocs = VOCS(
            **{
                "variables": {"x1": [0.0, 1.0]},
                "objectives": {"y1": "MAXIMIZE"},
                "constraints": {"c1": ["GREATER_THAN", 0.0]},
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

        assert torch.allclose(acq_values[0].double(), acq_values[1].double(), atol=5e-3)

        # if no values satisfy the constraint, EI should raise an error
        vocs_never_satisfied = VOCS(
            **{
                "variables": {"x1": [0.0, 1.0]},
                "objectives": {"y1": "MAXIMIZE"},
                "constraints": {"c1": ["GREATER_THAN", 2.0]},
            }
        )
        gen = ExpectedImprovementGenerator(vocs=vocs_never_satisfied)
        set_options(gen)
        gen.add_data(train_data)

        model = gen.train_model()
        with pytest.raises(
            RuntimeError,
            match="No feasible points found in the data; cannot compute expected improvement.",
        ):
            acq = gen.get_acquisition(model)

    def test_fixed_features_application(self):
        vocs = VOCS(
            variables={"x1": [0.0, 1.0], "x2": [0.0, 1.0]},
            objectives={"y1": "MINIMIZE"},
        )

        gen = ExpectedImprovementGenerator(vocs=vocs)
        gen.fixed_features = {"x1": 0.5}
        gen.add_data(
            pd.DataFrame(
                {
                    "x1": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    "x2": [0.1, 0.3, 0.5, 0.7, 0.9, 0.0],
                    "y1": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
                }
            )
        )

        # Mock model
        class MockModel:
            def posterior(self, X):
                return None

        model = MockModel()

        # Call get_acquisition and check if FixedFeatureAcquisitionFunction is applied
        acq = gen.get_acquisition(model)
        assert isinstance(acq, FixedFeatureAcquisitionFunction)
        assert acq.values.detach().item() == 0.5  # Fixed value for x1

    def test_get_acquisition_raises_value_error(self):
        gen = ExpectedImprovementGenerator(vocs=TEST_VOCS_BASE)
        with pytest.raises(ValueError, match="model cannot be None"):
            gen.get_acquisition(None)
