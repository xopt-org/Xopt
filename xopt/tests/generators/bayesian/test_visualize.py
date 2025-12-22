from matplotlib import pyplot as plt
import pytest
import numpy as np
import torch
import pandas as pd
from unittest.mock import MagicMock

from xopt.generators.bayesian import visualize
from xopt.generators.bayesian.upper_confidence_bound import (
    UpperConfidenceBoundGenerator,
)
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


def test__validate_names(vocs):
    outs, vars_ = visualize._validate_names(["z"], ["x"], vocs)
    assert outs == ["z"]
    assert vars_ == ["x"]


def test__get_reference_point(vocs, data):
    ref = visualize._get_reference_point(None, vocs, data)
    assert isinstance(ref, dict)
