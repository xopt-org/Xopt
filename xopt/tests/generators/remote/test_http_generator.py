import json
import socket
import threading
import time

import numpy as np
import pandas as pd
import pytest
import requests
import torch
import uvicorn

from gest_api.vocs import VOCS

from xopt.errors import XoptError
from xopt.generator import Generator
from xopt.generators.bayesian.bax.algorithms import GridOptimize
from xopt.generators.bayesian.bax_generator import BaxGenerator
from xopt.generators.bayesian.expected_improvement import ExpectedImprovementGenerator
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.random import RandomGenerator
from xopt.generators.remote.http import HTTPGenerator
from xopt.generators.remote.server import AUTH_TOKEN_ENV_VAR, app


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server_url():
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            requests.get(url + "/openapi.json", timeout=0.2)
            break
        except requests.ConnectionError:
            time.sleep(0.1)
    else:
        raise RuntimeError("test server did not start in time")

    yield url

    server.should_exit = True
    thread.join(timeout=5)


RANDOM_VOCS = VOCS(
    variables={"x1": [0.0, 1.0], "x2": [0.0, 1.0]},
    objectives={"y1": "MINIMIZE"},
)

BO_VOCS = VOCS(
    variables={"x": [0.0, 1.0]},
    objectives={"y": "MINIMIZE"},
)

# mirrors docs/examples/single_objective_bayes_opt/bax_tutorial.ipynb: BAX uses
# observables (no objectives), so it exercises `supports_no_objective`
BAX_VOCS = VOCS(
    variables={"x": [0, 2 * np.pi]},
    observables=["y1"],
)


class _UnregisteredGenerator(Generator):
    """Generator with a `name` that the server's registry doesn't recognize."""

    name = "not_a_real_generator"
    supports_single_objective: bool = True

    def generate(self, n_candidates) -> list[dict]:
        return []


class TestHTTPGeneratorRandom:
    def test_suggest_within_bounds(self, server_url):
        generator = HTTPGenerator(
            base_url=server_url, generator=RandomGenerator(vocs=RANDOM_VOCS)
        )
        points = generator.generate(5)
        assert len(points) == 5
        for point in points:
            assert 0.0 <= point["x1"] <= 1.0
            assert 0.0 <= point["x2"] <= 1.0
        generator.finalize()

    def test_ingest_updates_local_cache(self, server_url):
        generator = HTTPGenerator(
            base_url=server_url, generator=RandomGenerator(vocs=RANDOM_VOCS)
        )
        results = pd.DataFrame({"x1": [0.1, 0.2], "x2": [0.3, 0.4], "y1": [1.0, 2.0]})
        generator.ingest(results.to_dict(orient="records"))
        assert generator.data is not None
        assert len(generator.data) == 2
        generator.finalize()

    def test_full_replace_updates_remote_mirror(self, server_url):
        generator = HTTPGenerator(
            base_url=server_url, generator=RandomGenerator(vocs=RANDOM_VOCS)
        )
        # create a session first
        generator.generate(1)

        new_data = pd.DataFrame({"x1": [0.5], "x2": [0.5], "y1": [3.0]})
        generator.data = new_data

        mirror = generator.pull()
        assert len(mirror.data) == 1
        assert mirror.data.iloc[0]["x1"] == pytest.approx(0.5)
        generator.finalize()

    def test_unknown_generator_name_raises(self, server_url):
        generator = HTTPGenerator(
            base_url=server_url,
            generator=_UnregisteredGenerator(vocs=RANDOM_VOCS),
        )
        with pytest.raises(XoptError):
            generator.generate(1)

    def test_unknown_session_raises(self, server_url):
        generator = HTTPGenerator(
            base_url=server_url, generator=RandomGenerator(vocs=RANDOM_VOCS)
        )
        generator.generate(1)
        # force a bogus session id to simulate an expired/unknown session
        generator._session_id = "does-not-exist"
        with pytest.raises(XoptError):
            generator.generate(1)

    def test_auth_required(self, server_url, monkeypatch):
        monkeypatch.setenv(AUTH_TOKEN_ENV_VAR, "secret-token")

        unauthenticated = HTTPGenerator(
            base_url=server_url, generator=RandomGenerator(vocs=RANDOM_VOCS)
        )
        with pytest.raises(XoptError):
            unauthenticated.generate(1)

        authenticated = HTTPGenerator(
            base_url=server_url,
            generator=RandomGenerator(vocs=RANDOM_VOCS),
            api_key="secret-token",
        )
        points = authenticated.generate(1)
        assert len(points) == 1
        authenticated.finalize()

    def test_finalize_does_not_raise_when_unreachable(self):
        generator = HTTPGenerator(
            base_url="http://127.0.0.1:1",
            generator=RandomGenerator(vocs=RANDOM_VOCS),
        )
        generator._session_id = "whatever"
        with pytest.warns(UserWarning):
            generator.finalize()


class TestHTTPGeneratorExtraMethods:
    def test_visualize_model_passthrough(self, server_url):
        generator = HTTPGenerator(
            base_url=server_url,
            generator=ExpectedImprovementGenerator(vocs=BO_VOCS),
        )

        train_x = np.linspace(0.1, 0.9, 5)
        train_y = np.sin(2 * np.pi * train_x)
        data = pd.DataFrame({"x": train_x, "y": train_y})
        generator.ingest(data.to_dict(orient="records"))

        # triggers model training server-side
        generator.generate(1)

        fig, ax = generator.visualize_model()
        assert fig is not None
        assert ax is not None

        generator.finalize()

    def test_passthrough_reflects_latest_remote_state(self, server_url):
        generator = HTTPGenerator(
            base_url=server_url,
            generator=ExpectedImprovementGenerator(vocs=BO_VOCS),
        )

        train_x = np.linspace(0.1, 0.9, 5)
        train_y = np.sin(2 * np.pi * train_x)
        data = pd.DataFrame({"x": train_x, "y": train_y})
        generator.ingest(data.to_dict(orient="records"))

        generator.generate(1)
        # passthrough attribute access (not `.pull()`) must not cache a stale mirror
        assert len(generator.computation_time) == 1

        generator.generate(1)
        assert len(generator.computation_time) == 2

        generator.finalize()

    def test_mirror_serializes_with_http_generator(self, server_url):
        generator = HTTPGenerator(
            base_url=server_url,
            generator=ExpectedImprovementGenerator(vocs=BO_VOCS),
        )

        train_x = np.linspace(0.1, 0.9, 5)
        train_y = np.sin(2 * np.pi * train_x)
        data = pd.DataFrame({"x": train_x, "y": train_y})
        generator.ingest(data.to_dict(orient="records"))
        generator.generate(1)

        # triggers a pull(), replacing `generator.generator` with the fitted mirror
        generator.visualize_model()

        dumped = json.loads(
            generator.to_json(serialize_torch=True, serialize_inline=True)
        )
        assert dumped["generator"]["model"] is not None
        assert dumped["generator"]["computation_time"] is not None

        generator.finalize()


class TestHTTPGeneratorBax:
    def test_bax_workflow(self, server_url):
        # mirrors docs/examples/single_objective_bayes_opt/bax_tutorial.ipynb
        algorithm = GridOptimize(observable_names_ordered=["y1"], n_mesh_points=20)
        bax_generator = BaxGenerator(vocs=BAX_VOCS, algorithm=algorithm)
        bax_generator.gp_constructor.use_low_noise_prior = True

        generator = HTTPGenerator(base_url=server_url, generator=bax_generator)

        train_x = np.linspace(0.1, 2 * np.pi - 0.1, 3)
        data = pd.DataFrame({"x": train_x, "y1": np.sin(train_x)})
        generator.ingest(data.to_dict(orient="records"))

        # triggers model training + BAX algorithm execution server-side
        points = generator.generate(1)
        assert len(points) == 1
        assert 0 <= points[0]["x"] <= 2 * np.pi

        # BayesianGenerator.visualize_model still works through the passthrough
        # for a BAX-specific (no-objective, custom `algorithm` field) generator
        fig, ax = generator.visualize_model()
        assert fig is not None
        assert ax is not None

        generator.finalize()

        generator.finalize()


class ConstraintPrior(torch.nn.Module):
    """Prior mean function, mirrors docs/examples/single_objective_bayes_opt/custom_model.ipynb."""

    def forward(self, X):
        return (5.0 * torch.cos(2 * 3.14 * X + 0.25)).squeeze(dim=-1)


class TestHTTPGeneratorPriorMean:
    def test_custom_prior_mean_function(self, server_url):
        vocs = VOCS(
            variables={"x": [0.0, 1.0]},
            objectives={"y": "MAXIMIZE"},
            constraints={"c": ["LESS_THAN", 0.0]},
        )
        gp_constructor = StandardModelConstructor(
            mean_modules={"c": ConstraintPrior()}, use_low_noise_prior=True
        )
        bo_generator = ExpectedImprovementGenerator(
            vocs=vocs, gp_constructor=gp_constructor
        )

        generator = HTTPGenerator(base_url=server_url, generator=bo_generator)

        train_x = np.array([0.2, 0.5, 0.6])
        data = pd.DataFrame(
            {
                "x": train_x,
                "y": np.sin(2 * np.pi * train_x),
                "c": 5.0 * np.cos(2 * np.pi * train_x + 0.25),
            }
        )
        generator.ingest(data.to_dict(orient="records"))

        # triggers model training (with the custom prior mean) server-side
        points = generator.generate(1)
        assert len(points) == 1

        # extra-method passthrough pulls the fitted model back across the wire
        # (torch.save/pickle round trip), including the custom prior mean module
        fig, ax = generator.visualize_model()
        assert fig is not None
        assert ax is not None

        mirror = generator.generator
        assert isinstance(mirror.gp_constructor.mean_modules["c"], ConstraintPrior)

        generator.finalize()
