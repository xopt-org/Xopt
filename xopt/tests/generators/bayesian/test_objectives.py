import pytest
import torch
from torch import Tensor
from unittest.mock import MagicMock, patch
from xopt.generators.bayesian.objectives import feasibility
from xopt.generators.bayesian.objectives import create_constraint_callables
import functools


class DummyVOCS:
    def __init__(self, constraints=None, output_names=None):
        self.constraints = constraints
        self.output_names = output_names or []
        self.constraint_names = list(constraints.keys()) if constraints else []
        self.n_outputs = len(output_names) if output_names else 0
        self.n_objectives = len(output_names) if output_names else 0
        self.objectives = output_names or []


@pytest.fixture
def dummy_vocs():
    constraints = {"c1": ("LESS_THAN", 0.0), "c2": ("GREATER_THAN", 1.0)}
    output_names = ["c1", "c2", "y1"]
    return DummyVOCS(constraints=constraints, output_names=output_names)


@pytest.fixture
def dummy_vocs_no_constraints():
    output_names = ["y1", "y2"]
    return DummyVOCS(constraints=None, output_names=output_names)


@pytest.fixture
def dummy_model():
    model = MagicMock()
    dummy_posterior = MagicMock()
    model.posterior.return_value = dummy_posterior
    return model


@pytest.fixture
def dummy_sampler():
    def sampler(posterior):
        # Return tensor of shape [512, batch, q, m]
        return torch.ones(512, 2, 1, 3)

    return sampler


@pytest.fixture
def dummy_feasibility_objective():
    class DummyFeasibilityObjective:
        def __init__(self, constraints):
            pass

        def __call__(self, samples, X):
            # Return tensor of shape [512, 2, 1]
            return torch.ones(samples.shape[0], samples.shape[1], samples.shape[2])

    return DummyFeasibilityObjective


def test_feasibility_variants(
    dummy_vocs,
    dummy_vocs_no_constraints,
    dummy_model,
    dummy_sampler,
    dummy_feasibility_objective,
):
    # Basic feasibility
    X = torch.zeros(2, 1, 3)
    with (
        patch(
            "xopt.generators.bayesian.objectives.get_sampler",
            return_value=dummy_sampler,
        ),
        patch(
            "xopt.generators.bayesian.objectives.FeasibilityObjective",
            dummy_feasibility_objective,
        ),
    ):
        result = feasibility(X, dummy_model, dummy_vocs)
    assert isinstance(result, Tensor)
    assert result.shape == (2, 1)
    assert torch.all(result == 1.0)

    # No constraints
    X2 = torch.zeros(2, 1, 2)
    with (
        patch(
            "xopt.generators.bayesian.objectives.get_sampler",
            return_value=dummy_sampler,
        ),
        patch(
            "xopt.generators.bayesian.objectives.FeasibilityObjective",
            dummy_feasibility_objective,
        ),
    ):
        result2 = feasibility(X2, dummy_model, dummy_vocs_no_constraints)
    assert isinstance(result2, Tensor)
    assert result2.shape == (2, 1)
    assert torch.all(result2 == 1.0)

    # With posterior_transform
    X3 = torch.zeros(2, 1, 3)
    posterior_transform = MagicMock()
    with (
        patch(
            "xopt.generators.bayesian.objectives.get_sampler",
            return_value=dummy_sampler,
        ),
        patch(
            "xopt.generators.bayesian.objectives.FeasibilityObjective",
            dummy_feasibility_objective,
        ),
    ):
        result3 = feasibility(
            X3, dummy_model, dummy_vocs, posterior_transform=posterior_transform
        )
    assert isinstance(result3, Tensor)
    assert result3.shape == (2, 1)

    # Handles empty constraints
    vocs = DummyVOCS(constraints={}, output_names=["y1"])
    X4 = torch.zeros(2, 1, 1)
    with (
        patch(
            "xopt.generators.bayesian.objectives.get_sampler",
            return_value=dummy_sampler,
        ),
        patch(
            "xopt.generators.bayesian.objectives.FeasibilityObjective",
            dummy_feasibility_objective,
        ),
    ):
        result4 = feasibility(X4, dummy_model, vocs)
    assert isinstance(result4, Tensor)
    assert result4.shape == (2, 1)


def test_create_constraint_callables_less_than():
    class V:
        constraints = {"c1": ("LESS_THAN", 5.0)}
        output_names = ["c1"]
        constraint_names = ["c1"]

    vocs = V()
    callables = create_constraint_callables(vocs)
    assert isinstance(callables, list)
    assert len(callables) == 1
    # Should be positive if value < 5
    Z = torch.tensor([[4.0]])
    result = callables[0](Z)
    assert torch.allclose(result, torch.tensor([[-1.0]]))


def test_create_constraint_callables_greater_than():
    class V:
        constraints = {"c2": ("GREATER_THAN", 2.0)}
        output_names = ["c2"]
        constraint_names = ["c2"]

    vocs = V()
    callables = create_constraint_callables(vocs)
    assert isinstance(callables, list)
    assert len(callables) == 1
    # Should be negative if value < 2
    Z = torch.tensor([[1.0]])
    result = callables[0](Z)
    assert torch.allclose(result, torch.tensor([[1.0]]))


def test_create_constraint_callables_multiple():
    class V:
        constraints = {"c1": ("LESS_THAN", 0.0), "c2": ("GREATER_THAN", 1.0)}
        output_names = ["c1", "c2"]
        constraint_names = ["c1", "c2"]

    vocs = V()
    callables = create_constraint_callables(vocs)
    assert isinstance(callables, list)
    assert len(callables) == 2
    Z = torch.tensor([[0.0, 1.0]])
    res1 = callables[0](Z)
    res2 = callables[1](Z)
    assert res1.shape == (1,)
    assert res2.shape == (1,)


def test_create_constraint_callables_none():
    class V:
        constraints = None
        output_names = ["y1"]
        constraint_names = []

    vocs = V()
    callables = create_constraint_callables(vocs)
    assert callables is None


def test_create_constraint_callables_empty_dict():
    class V:
        constraints = {}
        output_names = ["y1"]
        constraint_names = []

    vocs = V()
    callables = create_constraint_callables(vocs)
    assert callables == []


def test_create_constraint_callables_partial_and_signature():
    class V:
        constraints = {"c1": ("LESS_THAN", 2.0), "c2": ("GREATER_THAN", -1.0)}
        output_names = ["c1", "c2"]
        constraint_names = ["c1", "c2"]

    vocs = V()
    callables = create_constraint_callables(vocs)

    # Check that each callable is a functools.partial object
    for cb in callables:
        assert isinstance(cb, functools.partial)
        # The underlying function should be named 'cbf'
        assert cb.func.__name__ == "cbf"
        # The partial should have the correct keywords set
        assert set(cb.keywords.keys()) == {"index", "value", "sign"}

    # Check that the callables behave as expected
    # c1: LESS_THAN 2.0, so sign=1, index=0, value=2.0
    # c2: GREATER_THAN -1.0, so sign=-1, index=1, value=-1.0
    Z = torch.tensor([[1.0, 0.0], [3.0, -2.0]])
    res1 = callables[0](Z)
    res2 = callables[1](Z)
    # For c1: 1*(Z[...,0] - 2.0)
    assert torch.allclose(res1, torch.tensor([-1.0, 1.0]))
    # For c2: -1*(Z[...,1] - (-1.0)) = -1*(Z[...,1] + 1.0)
    assert torch.allclose(res2, torch.tensor([-1.0, 1.0]))
