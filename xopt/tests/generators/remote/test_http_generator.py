"""HTTPGenerator remote integration and cloudpickle regression tests.

All 14 tests live in the original upstream regression-test location.

UPSTREAM HTTP TRANSPORT (12 tests)
    Uses the normal Xopt server in an independent Uvicorn subprocess.
    Covers HTTP sessions, data mirroring, authentication, random/UCB-related
    behavior, BAX, and an importable module-level PyTorch prior.

EXPERIMENTAL CLOUDPICKLE TRANSPORT (2 tests)
    Uses the opt-in cloudpickle server in a separate Uvicorn subprocess.
    The negative control checks that standard transport rejects a local
    (not importable) notebook-style PyTorch class. The positive control
    checks cloudpickle round-trip, candidate generation, and prior values.

Important scope: subprocesses run on the SAME machine and can share installed
packages. These tests do NOT prove cross-host Docker isolation. The next
integration milestone must run a backend without the scientist's source files.

Requirements for the two experimental tests:
    python -m pip install -e ./docs/examples/remote_cloudpickle

Run from repository root:
    python -m pytest xopt/tests/generators/remote/test_http_generator.py -v -s --no-cov

SECURITY: Never expose the cloudpickle deserialization endpoint to untrusted
clients; deserialization can execute arbitrary Python code.
"""

import json
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
import requests
import torch

from gest_api.vocs import VOCS

from xopt.errors import XoptError
from xopt.generator import Generator
from xopt.generators.bayesian.bax.algorithms import GridOptimize
from xopt.generators.bayesian.bax_generator import BaxGenerator
from xopt.generators.bayesian.expected_improvement import ExpectedImprovementGenerator
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.random import RandomGenerator
from xopt.generators.remote.http import HTTPGenerator
# Use an explicit name in the negative-control test so reviewers can see
# that it exercises upstream serialization, not the cloudpickle adapter.
from xopt.generators.remote.http import HTTPGenerator as UpstreamHTTPGenerator
from cloudpickle_xopt.http import HTTPGenerator as CloudpickleHTTPGenerator
from xopt.generators.remote.server import AUTH_TOKEN_ENV_VAR


def _free_port() -> int:
    """Ask the OS for an available localhost TCP port (not a fixed 8002)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def _uvicorn_subprocess(*, auth_token=None):
    """Start an isolated Python interpreter and reliably stop it afterward.

    This tests the same process boundary as running Uvicorn in another terminal.
    It does not require or reuse the manually launched server on port 8002.
    """
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    # Authentication belongs to the *server* environment, not pytest's.
    env.pop(AUTH_TOKEN_ENV_VAR, None)
    if auth_token is not None:
        env[AUTH_TOKEN_ENV_VAR] = auth_token

    command = [
        sys.executable, "-m", "uvicorn",
        "xopt.generators.remote.server:app",
        "--host", "127.0.0.1", "--port", str(port),
        "--log-level", "warning",
    ]
    process = subprocess.Popen(
        command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # Wait for a real HTTP response, not merely an open socket.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                error = process.stderr.read()
                raise RuntimeError(f"Uvicorn exited during startup:\n{error}")
            try:
                response = requests.get(url + "/openapi.json", timeout=0.3)
                response.raise_for_status()
                break
            except (requests.RequestException, OSError):
                time.sleep(0.1)
        else:
            raise RuntimeError(f"Uvicorn did not become ready at {url}")
        yield url
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stderr is not None:
            process.stderr.close()


@pytest.fixture(scope="module")
def server_url():
    """Normal upstream HTTP server, one independent process per test module."""
    with _uvicorn_subprocess() as url:
        yield url


@pytest.fixture
def authenticated_server_url():
    """Dedicated process with a token set *before* server startup."""
    with _uvicorn_subprocess(auth_token="secret-token") as url:
        yield url


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

    def test_auth_required(self, authenticated_server_url):
        server_url = authenticated_server_url

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


# ---------------------------------------------------------------------------
# Experimental cloudpickle integration tests (opt-in transport).
# REVIEW NOTE: Keep these beside the 12 original regressions so reviewers can
# compare unchanged upstream behavior against the opt-in serialization path.
# These tests use a different server app than the upstream regression tests.
# Both the client and server are independent processes, but on the same host.
# ---------------------------------------------------------------------------

def _cloudpickle_free_port():
    """Request an unused local TCP port from the operating system."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def _cloudpickle_server():
    """Launch the experimental app in a completely separate interpreter.

    This app contains BOTH ordinary upstream routes and opt-in cloudpickle
    routes, so the two serialization approaches face the same backend.
    """
    port = _cloudpickle_free_port()
    url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    # Do not inherit an unrelated auth token from the developer's shell.
    env.pop("XOPT_HTTP_AUTH_TOKEN", None)
    command = [
        sys.executable, "-m", "uvicorn", "cloudpickle_xopt.server:app",
        "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning",
    ]
    process = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited during startup:\n{process.stderr.read()}")
            try:
                response = requests.get(f"{url}/openapi.json", timeout=0.3)
                response.raise_for_status()
                break
            except requests.RequestException:
                time.sleep(0.1)
        else:
            raise RuntimeError(f"Server did not become ready at {url}")
        yield url
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stderr:
            process.stderr.close()


@pytest.fixture(scope="module")
def cloudpickle_server_url():
    with _cloudpickle_server() as url:
        yield url


def _notebook_generator():
    """Build a prior class *inside a function*, as with a notebook cell.

    A local class cannot be imported by its module-qualified name in the
    Uvicorn interpreter. This is different from Stage 1's module-level class.
    """
    amplitude = 5.0
    phase = 0.25

    class ConstraintPrior(torch.nn.Module):
        def forward(self, X):
            # c(x) = 5 cos(2*pi*x + 0.25); a periodic constraint prior.
            return (amplitude * torch.cos(2 * torch.pi * X + phase)).squeeze(-1)

    vocs = VOCS(
        variables={"x": [0.0, 1.0]},
        objectives={"y": "MAXIMIZE"},
        constraints={"c": ["LESS_THAN", 0.0]},
    )
    constructor = StandardModelConstructor(
        mean_modules={"c": ConstraintPrior()}, use_low_noise_prior=True
    )
    generator = ExpectedImprovementGenerator(vocs=vocs, gp_constructor=constructor)
    return generator, ConstraintPrior


def _training_data():
    """Three measurements of a sine objective and cosine constraint."""
    x = np.array([0.2, 0.5, 0.6])
    return pd.DataFrame({
        "x": x,
        "y": np.sin(2 * np.pi * x),
        "c": 5.0 * np.cos(2 * np.pi * x + 0.25),
    }).to_dict(orient="records")


def test_upstream_rejects_notebook_local_prior(cloudpickle_server_url):
    """Standard serialization must not silently lose the local class.

    Depending on Xopt/Pydantic internals, the failure can occur while the
    client serializes the generator or when the server decodes the request.
    Either is an expected limitation of the ordinary transport here.
    """
    local_generator, _ = _notebook_generator()
    remote = UpstreamHTTPGenerator(base_url=cloudpickle_server_url, generator=local_generator)
    try:
        with pytest.raises(Exception) as error:
            remote.ingest(_training_data())
            remote.generate(1)
        # A successful return would mean our negative control is no longer
        # valid and should be investigated, not marked as an expected failure.
        assert error.value is not None
    finally:
        remote.finalize()


def test_cloudpickle_round_trips_notebook_local_prior(cloudpickle_server_url):
    """Opt-in transport reconstructs and executes the actual custom class."""
    local_generator, prior_type = _notebook_generator()
    remote = CloudpickleHTTPGenerator(base_url=cloudpickle_server_url, generator=local_generator)
    try:
        remote.ingest(_training_data())
        points = remote.generate(1)
        assert len(points) == 1
        assert 0.0 <= points[0]["x"] <= 1.0

        # Fetch the server's generator, not the original local object.
        mirror = remote.pull()
        restored_prior = mirror.gp_constructor.mean_modules["c"]
        assert isinstance(restored_prior, prior_type)

        # Check the mathematical function, not merely the Python type name.
        X = torch.tensor([[0.2], [0.5]], dtype=torch.double)
        actual = restored_prior(X)
        expected = 5.0 * torch.cos(2 * torch.pi * X + 0.25).squeeze(-1)
        torch.testing.assert_close(actual, expected)
    finally:
        remote.finalize()
