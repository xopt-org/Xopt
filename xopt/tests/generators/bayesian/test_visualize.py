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
from xopt.vocs import ContextualVariable, VOCS


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


@pytest.fixture
def vocs_mixed_discrete():
    return VOCS(
        variables={"x": [0, 1], "y": {0.0, 5.0, 10.0}, "w": [0, 1]},
        objectives={"z": "MAXIMIZE"},
        constraints={},
        observables=["z"],
    )


@pytest.fixture
def data_mixed_discrete(vocs_mixed_discrete):
    return pd.DataFrame(
        {
            "x": [0.0, 0.5, 1.0, 0.25, 0.75],
            "y": [0.0, 5.0, 10.0, 5.0, 0.0],
            "w": [0.2, 0.2, 0.2, 0.2, 0.2],
            "z": [0.0, 0.4, 0.8, 0.2, 0.6],
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


def test_visualize_model_reference_title_visibility(vocs, data):
    generator = UpperConfidenceBoundGenerator(vocs=vocs)
    generator.add_data(data)
    generator.train_model()

    fig, _ = visualize.visualize_model(
        model=generator.model,
        vocs=vocs,
        data=data,
        tkwargs={},
        output_names=["z"],
        variable_names=["x"],
        n_grid=5,
        show_acquisition=False,
    )
    assert fig._suptitle is not None
    assert fig._suptitle.get_text().startswith("Reference point:\n")

    fig, ax = visualize.visualize_model(
        model=generator.model,
        vocs=vocs,
        data=data,
        tkwargs={},
        output_names=["z"],
        variable_names=["x", "y"],
        n_grid=5,
        show_acquisition=False,
    )
    assert fig._suptitle is None

    fig.suptitle("Temporary title")
    fig, _ = visualize.visualize_model(
        model=generator.model,
        vocs=vocs,
        data=data,
        tkwargs={},
        output_names=["z"],
        variable_names=["x", "y"],
        n_grid=5,
        show_acquisition=False,
        axes=ax,
    )
    assert fig._suptitle is None


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


def test_generate_input_mesh_mixed_discrete_shape_and_reference(vocs_mixed_discrete):
    mesh = visualize._generate_input_mesh(
        vocs=vocs_mixed_discrete,
        variable_names=["x", "y"],
        reference_point={"w": 0.2},
        data=None,
        n_grid=5,
        tkwargs={},
    )
    # Mixed mesh: 5 continuous points times 3 discrete choices.
    assert mesh.shape == (15, 3)
    assert np.allclose(mesh[:, 2].cpu().numpy(), 0.2)


def test_plot_model_prediction_1d_discrete_ticks(
    vocs_mixed_discrete, data_mixed_discrete
):
    model = DummyModel()
    ax = visualize.plot_model_prediction(
        model=model,
        vocs=vocs_mixed_discrete,
        data=data_mixed_discrete,
        tkwargs={},
        output_name="z",
        variable_names=["y"],
        n_grid=5,
    )
    assert ax is not None
    assert set(ax.get_xticks()) == {0.0, 5.0, 10.0}
    assert np.allclose(ax.get_xticks(), [0.0, 5.0, 10.0])


def test_plot_acquisition_function_1d_discrete_ticks(
    vocs_mixed_discrete, data_mixed_discrete
):
    acq = DummyAcq()
    ax = visualize.plot_acquisition_function(
        acquisition_function=acq,
        vocs=vocs_mixed_discrete,
        data=data_mixed_discrete,
        tkwargs={},
        variable_names=["y"],
        n_grid=5,
    )
    assert ax is not None
    assert set(ax.get_xticks()) == {0.0, 5.0, 10.0}
    assert np.allclose(ax.get_xticks(), [0.0, 5.0, 10.0])


def test_plot_acquisition_function_2d_mixed_discrete(
    vocs_mixed_discrete, data_mixed_discrete
):
    acq = DummyAcq()
    ax = visualize.plot_acquisition_function(
        acquisition_function=acq,
        vocs=vocs_mixed_discrete,
        data=data_mixed_discrete,
        tkwargs={},
        variable_names=["x", "y"],
        n_grid=5,
    )
    assert ax is not None


def test_visualize_model_with_fixed_variable_reference_point(
    vocs_mixed_discrete, data_mixed_discrete
):
    model = DummyModel()
    fig, ax = visualize.visualize_model(
        model=model,
        vocs=vocs_mixed_discrete,
        data=data_mixed_discrete,
        tkwargs={},
        output_names=["z"],
        variable_names=["x", "y"],
        reference_point={"w": 0.2},
        show_acquisition=False,
        n_grid=5,
    )
    assert fig is not None
    assert ax is not None


def test_visualize_model_requires_variable_selection_for_higher_dim(
    vocs_mixed_discrete, data_mixed_discrete
):
    model = DummyModel()
    with pytest.raises(ValueError):
        visualize.visualize_model(
            model=model,
            vocs=vocs_mixed_discrete,
            data=data_mixed_discrete,
            tkwargs={},
            output_names=["z"],
            variable_names=None,
            show_acquisition=False,
            n_grid=5,
        )


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


def test_plot_acquisition_function_with_contextual_axis_warns():
    vocs = VOCS(
        variables={"x": [0, 1], "context": ContextualVariable()},
        objectives={"z": "MAXIMIZE"},
        constraints={},
        observables=["z"],
    )
    data = pd.DataFrame(
        {
            "x": np.linspace(0, 1, 5),
            "context": np.linspace(0.2, 0.8, 5),
            "z": np.linspace(0.0, 1.0, 5),
        }
    )

    fig, axis = plt.subplots(1, 1)
    with pytest.warns(RuntimeWarning, match="Acquisition plot unavailable"):
        ax = visualize.plot_acquisition_function(
            acquisition_function=DummyAcq(),
            vocs=vocs,
            data=data,
            tkwargs={},
            variable_names=["x", "context"],
            n_grid=5,
            axis=axis,
        )

    assert any("Acquisition plot unavailable" in text.get_text() for text in ax.texts)
    plt.close(fig)


def test_visualize_model_with_contextual_axis_shows_acquisition_warning():
    vocs = VOCS(
        variables={"x": [0, 1], "context": ContextualVariable()},
        objectives={"z": "MAXIMIZE"},
        constraints={},
        observables=["z"],
    )
    data = pd.DataFrame(
        {
            "x": np.linspace(0, 1, 8),
            "context": np.linspace(0.1, 0.9, 8),
            "z": np.sin(np.linspace(0, np.pi, 8)),
        }
    )

    generator = UpperConfidenceBoundGenerator(vocs=vocs)
    generator.add_data(data)
    generator.train_model()

    with pytest.warns(RuntimeWarning, match="Acquisition plot unavailable"):
        fig, ax = generator.visualize_model(
            output_names=["z"],
            variable_names=["x", "context"],
            n_grid=5,
            show_acquisition=True,
        )

    acquisition_axis = ax[1, 0]
    assert any(
        "Acquisition plot unavailable" in text.get_text()
        for text in acquisition_axis.texts
    )
    plt.close(fig)


def test_visualize_model_contextual_slice_plots_acquisition():
    vocs = VOCS(
        variables={"x": [0, 1], "context": ContextualVariable()},
        objectives={"z": "MAXIMIZE"},
        constraints={},
        observables=["z"],
    )
    x = np.linspace(0.05, 0.95, 8)
    context = np.linspace(0.1, 0.9, 8)
    data = pd.DataFrame(
        {
            "x": x,
            "context": context,
            "z": -((x - context) ** 2),
        }
    )

    generator = UpperConfidenceBoundGenerator(vocs=vocs)
    generator.add_data(data)
    generator.train_model()

    fig, ax = generator.visualize_model(
        output_names=["z"],
        variable_names=["x"],
        n_grid=5,
        show_samples=False,
        show_acquisition=True,
    )

    acquisition_axis = ax[1, 0]
    assert not any(
        "Acquisition plot unavailable" in text.get_text()
        for text in acquisition_axis.texts
    )
    assert acquisition_axis.has_data()
    plt.close(fig)


def test_plot_model_prediction_contextual_axis_uses_explicit_domain_without_data_column():
    vocs = VOCS(
        variables={"x": [0, 1], "context": ContextualVariable(domain=[2.0, 3.0])},
        objectives={"z": "MAXIMIZE"},
        constraints={},
        observables=["z"],
    )
    data = pd.DataFrame(
        {
            "x": np.linspace(0, 1, 5),
            "z": np.linspace(0.0, 1.0, 5),
        }
    )

    ax = visualize.plot_model_prediction(
        model=DummyModel(),
        vocs=vocs,
        output_name="z",
        data=data,
        tkwargs={},
        variable_names=["context"],
        reference_point={"x": 0.5, "context": 2.5},
        n_grid=5,
        show_samples=False,
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
    assert ref == {"x": 1.0, "y": 1.0}


def test_get_reference_point_returns_explicit_reference(vocs, data):
    explicit = {"x": 0.123, "y": 0.456}
    ref = visualize._get_reference_point(explicit, vocs, data)
    assert ref == explicit


def test_get_reference_point_falls_back_to_idx_for_multi_objective():
    vocs = VOCS(
        variables={"x": [0, 1], "y": [0, 1]},
        objectives={"obj1": "MAXIMIZE", "obj2": "MINIMIZE"},
        constraints={},
        observables=["obj1", "obj2"],
    )
    data = pd.DataFrame(
        {
            "x": [0.1, 0.9],
            "y": [0.2, 0.8],
            "obj1": [1.0, 0.0],
            "obj2": [1.0, 0.0],
        }
    )

    ref = visualize._get_reference_point(None, vocs, data, idx=0)
    assert ref == {"x": 0.1, "y": 0.2}


def test_get_reference_point_falls_back_to_idx_for_explore_objective():
    vocs = VOCS(
        variables={"x": [0, 1], "y": [0, 1]},
        objectives={"z": "EXPLORE"},
        constraints={},
        observables=["z"],
    )
    data = pd.DataFrame(
        {
            "x": [0.1, 0.9],
            "y": [0.2, 0.8],
            "z": [1.0, 0.0],
        }
    )

    ref = visualize._get_reference_point(None, vocs, data, idx=0)
    assert ref == {"x": 0.1, "y": 0.2}


def test_get_reference_point_falls_back_to_idx_for_no_feasible_points():
    vocs = VOCS(
        variables={"x": [0, 1], "y": [0, 1]},
        objectives={"z": "MAXIMIZE"},
        constraints={"z": ["LESS_THAN", -10.0]},
        observables=["z"],
    )
    data = pd.DataFrame(
        {
            "x": [0.2, 0.8],
            "y": [0.3, 0.9],
            "z": [0.1, 0.2],
        }
    )

    ref = visualize._get_reference_point(None, vocs, data, idx=0)
    assert ref == {"x": 0.2, "y": 0.3}


def test_get_reference_point_falls_back_to_idx_for_runtime_error(
    vocs, data, monkeypatch
):
    def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("synthetic select_best failure")

    monkeypatch.setattr(visualize, "select_best", raise_runtime_error)
    ref = visualize._get_reference_point(None, vocs, data, idx=0)
    assert ref == {"x": 0.0, "y": 0.0}


def test_format_reference_point_title_wraps_long_text():
    reference_point = {
        "SIOC:SYS1:ML00:AO551": 0.59,
        "SIOC:SYS1:ML00:AO556": -1.2,
        "SIOC:SYS1:ML00:AO501": 0.48,
        "SIOC:SYS1:ML00:AO506": 0.42,
    }
    title = visualize._format_reference_point_title(
        reference_point,
        list(reference_point.keys()),
        max_line_length=45,
    )

    assert title.startswith("Reference point:\n")
    assert title.count("\n") > 1


def test_format_reference_point_title_empty_without_names():
    title = visualize._format_reference_point_title({}, [])
    assert title == "Reference point"


def test_format_reference_point_title_handles_empty_wrap_result(monkeypatch):
    reference_point = {"x": 0.25}

    def empty_wrap(*args, **kwargs):
        return []

    monkeypatch.setattr(visualize.textwrap, "wrap", empty_wrap)
    title = visualize._format_reference_point_title(
        reference_point,
        ["x"],
        max_line_length=1,
    )
    assert title.startswith("Reference point:\n")
    assert "x: 0.25" in title


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
