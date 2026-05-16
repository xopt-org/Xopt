from copy import deepcopy
import time

import pandas as pd
import pytest
import torch

from gest_api.vocs import VOCS

from xopt.generators.bayesian.bax.algorithms import GridOptimize
from xopt.generators.bayesian.bax_generator import BaxGenerator
from xopt.generators.bayesian.bayesian_exploration import BayesianExplorationGenerator
from xopt.generators.bayesian.expected_improvement import ExpectedImprovementGenerator
from xopt.generators.bayesian.expected_improvement import TDExpectedImprovementGenerator
from xopt.generators.bayesian.mobo import MOBOGenerator
from xopt.generators.bayesian.multi_fidelity import MultiFidelityGenerator
from xopt.generators.bayesian.upper_confidence_bound import (
    UpperConfidenceBoundGenerator,
)
from xopt.generators.bayesian.upper_confidence_bound import (
    TDUpperConfidenceBoundGenerator,
)


def _build_training_data(discrete_only: bool) -> pd.DataFrame:
    if discrete_only:
        x1 = [0.0, 1.0, 0.0, 1.0]
    else:
        x1 = [0.1, 0.3, 0.7, 0.9]

    x2 = [0.0, 5.0, 10.0, 5.0]

    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "y1": x2,
            "c1": x1,
        }
    )


def _build_mobo_training_data(discrete_only: bool) -> pd.DataFrame:
    if discrete_only:
        x1 = [0.0, 1.0, 0.0, 1.0]
    else:
        x1 = [0.1, 0.3, 0.7, 0.9]

    x2 = [0.0, 5.0, 10.0, 5.0]

    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "y1": x2,
            "y2": x1,
        }
    )


def _build_time_dependent_training_data(discrete_only: bool) -> pd.DataFrame:
    data = _build_training_data(discrete_only)
    # Keep timestamps slightly in the past and monotonic.
    now = time.time()
    data["time"] = [now - 0.05, now - 0.03, now - 0.02, now - 0.01]
    return data


def _build_multi_fidelity_training_data(discrete_only: bool) -> pd.DataFrame:
    if discrete_only:
        x1 = [0.0, 1.0, 0.0, 1.0]
    else:
        x1 = [0.1, 0.3, 0.7, 0.9]

    x2 = [0.0, 5.0, 10.0, 5.0]
    s = [0.2, 0.4, 0.7, 1.0]

    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "s": s,
            "y1": x2,
        }
    )


def _assert_discrete_membership(candidates, discrete_sets):
    for candidate in candidates:
        for name, allowed in discrete_sets.items():
            assert float(candidate[name]) in allowed


@pytest.mark.parametrize("discrete_only", [False, True])
@pytest.mark.parametrize("generator_name", ["ei", "exploration", "ucb", "bax"])
def test_discrete_candidate_generation_across_generators(generator_name, discrete_only):
    torch.manual_seed(7)

    if discrete_only:
        variables = {"x1": {0.0, 1.0}, "x2": {0.0, 5.0, 10.0}}
    else:
        variables = {"x1": [0.0, 1.0], "x2": {0.0, 5.0, 10.0}}

    if generator_name == "ei":
        vocs = VOCS(
            variables=variables,
            objectives={"y1": "MINIMIZE"},
            constraints={"c1": ["GREATER_THAN", -0.1]},
        )
        gen = ExpectedImprovementGenerator(vocs=vocs)
    elif generator_name == "exploration":
        vocs = VOCS(
            variables=variables,
            objectives={"y1": "EXPLORE"},
            constraints={},
        )
        gen = BayesianExplorationGenerator(vocs=vocs)
    elif generator_name == "ucb":
        vocs = VOCS(
            variables=variables,
            objectives={"y1": "MINIMIZE"},
            constraints={},
        )
        gen = UpperConfidenceBoundGenerator(vocs=vocs)
    elif generator_name == "bax":
        vocs = VOCS(
            variables=variables,
            objectives={},
            observables=["y1"],
            constraints={},
        )
        gen = BaxGenerator(vocs=vocs, algorithm=GridOptimize())
    else:
        raise ValueError(f"Unsupported generator case: {generator_name}")

    # Keep integration tests lightweight while still exercising full candidate flow.
    gen.numerical_optimizer.n_restarts = 1
    gen.n_monte_carlo_samples = 4
    if hasattr(gen.numerical_optimizer, "mixed_max_discrete_configurations"):
        gen.numerical_optimizer.mixed_max_discrete_configurations = 64
        gen.numerical_optimizer.discrete_max_choices = 128

    training_data = _build_training_data(discrete_only)
    gen.add_data(deepcopy(training_data))

    candidates = gen.generate(1)
    assert len(candidates) == 1

    discrete_sets = {"x2": {0.0, 5.0, 10.0}}
    if discrete_only:
        discrete_sets["x1"] = {0.0, 1.0}
    else:
        assert 0.0 <= float(candidates[0]["x1"]) <= 1.0

    _assert_discrete_membership(candidates, discrete_sets)


@pytest.mark.parametrize("discrete_only", [False, True])
def test_mobo_discrete_candidate_generation(discrete_only):
    torch.manual_seed(7)

    if discrete_only:
        variables = {"x1": {0.0, 1.0}, "x2": {0.0, 5.0, 10.0}}
    else:
        variables = {"x1": [0.0, 1.0], "x2": {0.0, 5.0, 10.0}}

    vocs = VOCS(
        variables=variables,
        objectives={"y1": "MINIMIZE", "y2": "MINIMIZE"},
        constraints={},
    )

    gen = MOBOGenerator(vocs=vocs, reference_point={"y1": 12.0, "y2": 1.5})
    gen.numerical_optimizer.n_restarts = 1
    gen.n_monte_carlo_samples = 4
    if hasattr(gen.numerical_optimizer, "mixed_max_discrete_configurations"):
        gen.numerical_optimizer.mixed_max_discrete_configurations = 64
        gen.numerical_optimizer.discrete_max_choices = 128

    gen.add_data(_build_mobo_training_data(discrete_only))
    candidates = gen.generate(1)
    assert len(candidates) == 1

    discrete_sets = {"x2": {0.0, 5.0, 10.0}}
    if discrete_only:
        discrete_sets["x1"] = {0.0, 1.0}
    else:
        assert 0.0 <= float(candidates[0]["x1"]) <= 1.0

    _assert_discrete_membership(candidates, discrete_sets)


@pytest.mark.parametrize("discrete_only", [False, True])
@pytest.mark.parametrize("generator_name", ["td_ei", "td_ucb"])
def test_time_dependent_discrete_candidate_generation(generator_name, discrete_only):
    torch.manual_seed(7)

    if discrete_only:
        variables = {"x1": {0.0, 1.0}, "x2": {0.0, 5.0, 10.0}}
    else:
        variables = {"x1": [0.0, 1.0], "x2": {0.0, 5.0, 10.0}}

    vocs = VOCS(
        variables=variables,
        objectives={"y1": "MINIMIZE"},
        constraints={},
    )

    if generator_name == "td_ei":
        gen = TDExpectedImprovementGenerator(vocs=vocs)
    elif generator_name == "td_ucb":
        gen = TDUpperConfidenceBoundGenerator(vocs=vocs)
    else:
        raise ValueError(f"Unsupported generator case: {generator_name}")

    gen.added_time = 1e-4
    gen.numerical_optimizer.n_restarts = 1
    gen.n_monte_carlo_samples = 4
    if hasattr(gen.numerical_optimizer, "mixed_max_discrete_configurations"):
        gen.numerical_optimizer.mixed_max_discrete_configurations = 64
        gen.numerical_optimizer.discrete_max_choices = 128

    gen.add_data(_build_time_dependent_training_data(discrete_only))
    candidates = gen.generate(1)
    assert len(candidates) == 1

    discrete_sets = {"x2": {0.0, 5.0, 10.0}}
    if discrete_only:
        discrete_sets["x1"] = {0.0, 1.0}
    else:
        assert 0.0 <= float(candidates[0]["x1"]) <= 1.0

    _assert_discrete_membership(candidates, discrete_sets)


@pytest.mark.parametrize("discrete_only", [False, True])
def test_multi_fidelity_discrete_candidate_generation(discrete_only):
    torch.manual_seed(7)

    if discrete_only:
        variables = {"x1": {0.0, 1.0}, "x2": {0.0, 5.0, 10.0}}
    else:
        variables = {"x1": [0.0, 1.0], "x2": {0.0, 5.0, 10.0}}

    vocs = VOCS(
        variables=variables,
        objectives={"y1": "MINIMIZE"},
        constraints={},
    )

    gen = MultiFidelityGenerator(vocs=vocs)
    gen.numerical_optimizer.n_restarts = 1
    gen.n_monte_carlo_samples = 4
    if hasattr(gen.numerical_optimizer, "mixed_max_discrete_configurations"):
        gen.numerical_optimizer.mixed_max_discrete_configurations = 64
        gen.numerical_optimizer.discrete_max_choices = 128

    gen.add_data(_build_multi_fidelity_training_data(discrete_only))
    candidates = gen.generate(1)
    assert len(candidates) == 1

    discrete_sets = {"x2": {0.0, 5.0, 10.0}}
    if discrete_only:
        discrete_sets["x1"] = {0.0, 1.0}
    else:
        assert 0.0 <= float(candidates[0]["x1"]) <= 1.0

    # Multi-fidelity adds a continuous fidelity variable `s` in [0, 1].
    assert 0.0 <= float(candidates[0]["s"]) <= 1.0

    _assert_discrete_membership(candidates, discrete_sets)
