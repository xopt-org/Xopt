import torch
from botorch.models import ModelListGP, SingleTaskGP

from xopt.generators.bayesian.objectives import create_constraint_callables, feasibility
from xopt.vocs import VOCS


class TestConstraints:
    def test_create_constraint_callables(self):
        vocs = VOCS(
            variables={"x": [0, 1]},
            objectives={"y": "MAXIMIZE"},
            constraints={"c": ["LESS_THAN", 10]},
        )
        f = create_constraint_callables(vocs)

        # test a values
        test_value = torch.tensor([10.0, -10.0]).reshape(1, 1, 2)
        assert f[0](test_value).float() == -20

        test_value = torch.tensor([10.0, 10.0]).reshape(1, 1, 2)
        assert f[0](test_value).float() == 0

        test_value = torch.tensor([10.0, 100.0]).reshape(1, 1, 2)
        assert f[0](test_value).float() == 90

        vocs = VOCS(
            variables={"x": [0, 1]},
            objectives={"y": "MAXIMIZE"},
            constraints={"c": ["GREATER_THAN", 10]},
        )
        f = create_constraint_callables(vocs)

        # test values
        test_value = torch.tensor([10.0, -10.0]).reshape(1, 1, 2)
        assert f[0](test_value).float() == 20

        test_value = torch.tensor([10.0, 10.0]).reshape(1, 1, 2)
        assert f[0](test_value).float() == 0

        test_value = torch.tensor([10.0, 100.0]).reshape(1, 1, 2)
        assert f[0](test_value).float() == -90

    def test_w_model(self):
        train_x = torch.zeros(1).reshape(1, 1, 1)
        train_y = torch.zeros(1).reshape(1, 1, 1)
        test_x = torch.linspace(0, 1, 25).reshape(-1, 1, 1)

        gp = ModelListGP(SingleTaskGP(train_x, train_y), SingleTaskGP(train_x, train_y))

        vocs = VOCS(
            variables={"x": [0, 1]},
            objectives={"y": "MAXIMIZE"},
            constraints={"c": ["LESS_THAN", 0]},
        )
        feas = feasibility(test_x, gp, vocs)
        assert torch.allclose(feas.flatten(), 0.5 * torch.ones(25), rtol=1e-2)

        gp = ModelListGP(SingleTaskGP(train_x, train_y), SingleTaskGP(train_x, train_y))

        vocs = VOCS(
            variables={"x": [0, 1]},
            objectives={"y": "MAXIMIZE"},
            constraints={"c": ["LESS_THAN", -1]},
        )
        feas = feasibility(test_x, gp, vocs)
        assert torch.allclose(feas.flatten(), 0.1 * torch.ones(25), atol=1e-1)

        gp = ModelListGP(SingleTaskGP(train_x, train_y), SingleTaskGP(train_x, train_y))

        vocs = VOCS(
            variables={"x": [0, 1]},
            objectives={"y": "MAXIMIZE"},
            constraints={"c": ["GREATER_THAN", -1]},
        )
        feas = feasibility(test_x, gp, vocs)
        assert torch.allclose(feas.flatten(), 0.9 * torch.ones(25), atol=1e-1)
