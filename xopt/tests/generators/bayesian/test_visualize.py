from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
from matplotlib import pyplot as plt

from xopt import Evaluator, Xopt
from xopt.generators.bayesian import MOBOGenerator, visualize
from xopt.generators.bayesian.upper_confidence_bound import (
    UpperConfidenceBoundGenerator,
)
from xopt.resources.test_functions.tnk import evaluate_TNK, tnk_vocs
from xopt.vocs import VOCS


class DummyPosterior:
    def __init__(self, shape):
        self.mean = torch.zeros(*shape, 1)
        self.mvn = MagicMock(variance=torch.ones(*shape, 1))


class DummyModel:
    def __init__(self):
        self.models = [self]
        self.input_transform = MagicMock()
        self.mean_module = MagicMock()
        self.outcome_transform = MagicMock()

    def posterior(self, X, **kwargs):
        # X: torch.Tensor of shape (n, d)
        n = X.shape[0]
        return DummyPosterior((n,))

    def __call__(self, x):
        n = x.shape[0]
        return MagicMock(mean=torch.zeros(n, 1), variance=torch.ones(n, 1))


class DummyAcq:
    def __call__(self, x):
        return torch.zeros(x.shape[0], 1)

    def base_acquisition(self, x):
        return torch.ones(x.shape[0], 1)


@pytest.fixture
def vocs():
    return VOCS(
        variables={"x": [0, 1], "y": [0, 1]},
        objectives={"z": "MAXIMIZE"},
        constraints={},
        observables=["z"],
    )


@pytest.fixture
def data(vocs):
    return pd.DataFrame(
        {
            "x": np.linspace(0, 1, 5),
            "y": np.linspace(0, 1, 5),
            "z": np.linspace(0, 1, 5),
        }
    )


@pytest.mark.parametrize("show_acquisition", [True, False])
@pytest.mark.parametrize("variable_names", [["x"], ["x", "y"]])
def test_visualize_model(vocs, data, variable_names, show_acquisition):
    generator = UpperConfidenceBoundGenerator(vocs=vocs)
    generator.add_data(data)
    generator.train_model()

    fig, ax = visualize.visualize_model(
        model=generator.model,
        vocs=vocs,
        data=data,
        tkwargs={},
        output_names=["z"],
        variable_names=variable_names,
        n_grid=5,
        show_acquisition=show_acquisition,
    )
    assert ax is not None

    fig, ax = visualize.visualize_model(
        model=generator.model,
        vocs=vocs,
        data=data,
        tkwargs={},
        output_names=["z"],
        variable_names=variable_names,
        n_grid=5,
        show_acquisition=show_acquisition,
        axes=ax,
    )


def test_plot_model_prediction_1d(vocs, data):
    model = DummyModel()
    ax = visualize.plot_model_prediction(
        model=model,
        vocs=vocs,
        data=data,
        tkwargs={},
        output_name="z",
        variable_names=["x"],
        n_grid=5,
    )
    assert ax is not None


def test_plot_model_prediction_2d(vocs, data):
    model = DummyModel()
    ax = visualize.plot_model_prediction(
        model=model,
        vocs=vocs,
        data=data,
        tkwargs={},
        output_name="z",
        variable_names=["x", "y"],
        n_grid=5,
    )
    assert ax is not None


def test_plot_acquisition_function_1d(vocs, data):
    acq = DummyAcq()
    ax = visualize.plot_acquisition_function(
        acquisition_function=acq,
        vocs=vocs,
        data=data,
        tkwargs={},
        variable_names=["x"],
        n_grid=5,
    )
    assert ax is not None


def test_plot_acquisition_function_2d(vocs, data):
    acq = DummyAcq()
    ax = visualize.plot_acquisition_function(
        acquisition_function=acq,
        vocs=vocs,
        data=data,
        tkwargs={},
        variable_names=["x", "y"],
        n_grid=5,
    )
    assert ax is not None


# @pytest.fixture(autouse=True)
# def patch_get_sampler():
#     with patch("xopt.generators.bayesian.visualize.get_sampler") as mock_get_sampler:
#         def dummy_sampler(*args, **kwargs):
#             class DummySampler:
#                 def __call__(self, posterior, sample_shape=torch.Size(), base_samples=None):
#                     n = posterior.mean.shape[0]
#                     return torch.zeros(sample_shape.numel() or 1, n, 1)
#             return DummySampler()
#         mock_get_sampler.side_effect = dummy_sampler
#         yield
#
# def test_plot_feasibility_1d(vocs, data):
#     model = DummyModel()
#     ax = visualize.plot_feasibility(
#         model=model,
#         vocs=vocs,
#         data=data,
#         tkwargs={},
#         variable_names=["x"],
#         n_grid=5
#     )
#     assert ax is not None
#
# def test_plot_feasibility_2d(vocs, data):
#     model = DummyModel()
#     ax = visualize.plot_feasibility(
#         model=model,
#         vocs=vocs,
#         data=data,
#         tkwargs={},
#         variable_names=["x", "y"],
#         n_grid=5
#     )
#     assert ax is not None


def test_plot_samples_1d(vocs, data):
    ax = visualize.plot_samples(
        vocs=vocs, data=data, output_name="z", variable_names=["x"]
    )
    assert ax is not None


def test_plot_samples_2d(vocs, data):
    ax = visualize.plot_samples(
        vocs=vocs, data=data, output_name="z", variable_names=["x", "y"]
    )
    assert ax is not None


def test_validate_names(vocs):
    outs, vars_ = visualize._validate_names(["z"], ["x"], vocs)
    assert outs == ["z"]
    assert vars_ == ["x"]

    with pytest.raises(ValueError):
        visualize._validate_names(["z"], ["x", "m", "n"], vocs)

    with pytest.raises(ValueError):
        visualize._validate_names(["z"], ["a"], vocs)


def test_get_reference_point(vocs, data):
    ref = visualize._get_reference_point(None, vocs, data)
    assert isinstance(ref, dict)


def test_verify_axes():
    with pytest.raises(ValueError):
        visualize._verify_axes("str", 5, 5)

    fig, ax = plt.subplots(2, 2)
    visualize._verify_axes(ax, 2, 2)
    with pytest.raises(ValueError):
        visualize._verify_axes(ax, 3, 2)

    fig, ax = plt.subplots(1, 3)
    visualize._verify_axes(ax, 1, 3)
    with pytest.raises(ValueError):
        visualize._verify_axes(ax, 1, 2)

    fig, ax = plt.subplots()
    visualize._verify_axes(ax, 1, 1)


def test_get_figure_from_axes():
    fig, ax = plt.subplots()
    returned_fig = visualize._get_figure_from_axes(ax)
    assert returned_fig is fig

    with pytest.raises(ValueError):
        visualize._get_figure_from_axes(None)

    with pytest.raises(ValueError):
        visualize._get_figure_from_axes("str")

    with pytest.raises(ValueError):
        visualize._get_figure_from_axes(np.array([1, 2, 3]))


def test_get_axis():
    assert isinstance(visualize._get_axis(None), plt.Axes)
    fig, ax = plt.subplots()
    assert visualize._get_axis(ax) is ax

    with pytest.raises(ValueError):
        visualize._get_axis("str")


def test_in_generator():
    evaluator = Evaluator(function=evaluate_TNK)

    generator = MOBOGenerator(vocs=tnk_vocs, reference_point={"y1": 1.5, "y2": 1.5})
    generator.n_monte_carlo_samples = 1
    generator.numerical_optimizer.n_restarts = 1
    generator.numerical_optimizer.max_iter = 1
    generator.gp_constructor.use_low_noise_prior = True

    X = Xopt(generator=generator, evaluator=evaluator)

    with pytest.raises(ValueError):
        X.generator.visualize_model()

    X.evaluate_data(pd.DataFrame({"x1": [1.0, 0.75], "x2": [0.75, 1.0]}))

    X.step()

    X.generator.visualize_model(
        show_acquisition=True, show_feasibility=True, show_prior_mean=True
    )
    X.generator.visualize_model(
        variable_names=["x1"],
        show_acquisition=True,
        show_feasibility=True,
        show_prior_mean=True,
    )
