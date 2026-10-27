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
    python -m pip install -e ".[remote]"

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
from xopt.generators.remote.cloudpickle import CloudpickleHTTPGenerator
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
        sys.executable,
        "-m",
        "uvicorn",
        "xopt.generators.remote.server:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
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
        fractional_vocs = VOCS(
            variables={"x": [0.25, 2 * np.pi - 0.25]},
            observables=["y1"],
        )
        bax_generator = BaxGenerator(vocs=fractional_vocs, algorithm=algorithm)
        bax_generator.gp_constructor.use_low_noise_prior = True

        generator = HTTPGenerator(base_url=server_url, generator=bax_generator)

        # Verify fractional BAX bounds survive the HTTP round-trip.
        remote_generator = generator.generator

        original_bounds = bax_generator.vocs.variables["x"].domain
        remote_bounds = remote_generator.vocs.variables["x"].domain

        np.testing.assert_allclose(
            remote_bounds,
            original_bounds,
            rtol=0,
            atol=1e-12,
        )

        assert remote_bounds[0] == 0.25
        assert remote_bounds[1] == 2 * np.pi - 0.25

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
    env["XOPT_HTTP_GENERATOR_TOKEN"] = "cloudpickle-test-token"
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "xopt.generators.remote.cloudpickle_server:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    process = subprocess.Popen(
        command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Server exited during startup:\n{process.stderr.read()}"
                )
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
    return pd.DataFrame(
        {
            "x": x,
            "y": np.sin(2 * np.pi * x),
            "c": 5.0 * np.cos(2 * np.pi * x + 0.25),
        }
    ).to_dict(orient="records")


def test_upstream_rejects_notebook_local_prior(cloudpickle_server_url):
    """Standard serialization must not silently lose the local class.

    Depending on Xopt/Pydantic internals, the failure can occur while the
    client serializes the generator or when the server decodes the request.
    Either is an expected limitation of the ordinary transport here.
    """
    local_generator, _ = _notebook_generator()
    remote = UpstreamHTTPGenerator(
        base_url=cloudpickle_server_url,
        generator=local_generator,
        api_key="cloudpickle-test-token",
    )
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
    remote = CloudpickleHTTPGenerator(
        base_url=cloudpickle_server_url,
        generator=local_generator,
        api_key="cloudpickle-test-token",
    )
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


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required for cross-device snapshot test",
)
def test_cloudpickle_cuda_snapshot_preserves_live_generator():  # pragma: no cover
    """Export CUDA tensors to CPU without modifying the live generator."""
    import base64

    import cloudpickle

    from xopt.generators.remote import server as upstream
    from xopt.generators.remote.cloudpickle_server import get_cloudpickle_state

    class CUDAState:
        def __init__(self):
            self.model = torch.nn.Linear(2, 1).cuda()
            self.algorithm_results = {
                "input_execution_paths": torch.ones(3, 2, device="cuda"),
                "output_execution_paths": torch.ones(3, 1, device="cuda"),
            }

    live = CUDAState()
    session_id = "cuda-snapshot-regression"

    with upstream._lock:
        upstream._sessions[session_id] = upstream.GeneratorSession(generator=live)

    try:
        response = get_cloudpickle_state(session_id)
        snapshot = cloudpickle.loads(base64.b64decode(response["payload"]))

        assert next(snapshot.model.parameters()).device.type == "cpu"
        assert snapshot.algorithm_results["input_execution_paths"].device.type == "cpu"
        assert snapshot.algorithm_results["output_execution_paths"].device.type == "cpu"

        assert next(live.model.parameters()).device.type == "cuda"
        assert live.algorithm_results["input_execution_paths"].device.type == "cuda"
        assert live.algorithm_results["output_execution_paths"].device.type == "cuda"

        assert snapshot is not live
    finally:
        with upstream._lock:
            upstream._sessions.pop(session_id, None)


def test_cloudpickle_cpu_snapshot_preserves_live_generator():
    """A CPU snapshot must not share mutable state with the live generator."""
    import base64

    import cloudpickle

    from xopt.generators.remote import server as upstream
    from xopt.generators.remote.cloudpickle_server import get_cloudpickle_state

    class CPUState:
        def __init__(self):
            self.model = torch.nn.Linear(2, 1)
            self.algorithm_results = {
                "input_execution_paths": torch.ones(3, 2),
            }

    live = CPUState()
    session_id = "cpu-snapshot-regression"

    with upstream._lock:
        upstream._sessions[session_id] = upstream.GeneratorSession(generator=live)

    try:
        response = get_cloudpickle_state(session_id)
        snapshot = cloudpickle.loads(base64.b64decode(response["payload"]))

        assert snapshot is not live
        assert snapshot.model is not live.model
        assert next(snapshot.model.parameters()).device.type == "cpu"

        snapshot.algorithm_results["input_execution_paths"].zero_()
        assert torch.all(live.algorithm_results["input_execution_paths"] == 1)

        with torch.no_grad():
            next(snapshot.model.parameters()).zero_()

        assert not torch.all(next(live.model.parameters()) == 0)
    finally:
        with upstream._lock:
            upstream._sessions.pop(session_id, None)


def test_cloudpickle_snapshot_cuda_preference_falls_back_to_cpu():
    """A CPU snapshot retains CUDA preference but runs on CPU without CUDA."""
    import base64
    import warnings
    from unittest.mock import patch

    import cloudpickle

    from xopt.generators.bayesian.upper_confidence_bound import (
        UpperConfidenceBoundGenerator,
    )
    from xopt.generators.remote import server as upstream
    from xopt.generators.remote.cloudpickle_server import get_cloudpickle_state
    from xopt.vocs import VOCS

    vocs = VOCS(
        variables={"x": [0.0, 1.0]},
        objectives={"y": "MINIMIZE"},
    )
    live = UpperConfidenceBoundGenerator(vocs=vocs, use_cuda=True)
    session_id = "cpu-cuda-fallback-regression"

    with upstream._lock:
        upstream._sessions[session_id] = upstream.GeneratorSession(generator=live)

    try:
        response = get_cloudpickle_state(session_id)
        snapshot = cloudpickle.loads(base64.b64decode(response["payload"]))

        assert snapshot is not live
        assert snapshot.use_cuda is True

        with patch.object(torch.cuda, "is_available", return_value=False):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                options = snapshot.tkwargs

        assert options["device"] == "cpu"
        assert options["dtype"] == torch.double
        assert any("Using CPU instead" in str(w.message) for w in caught)

        assert live.use_cuda is True
    finally:
        with upstream._lock:
            upstream._sessions.pop(session_id, None)


def test_http_generator_validator_preserves_non_generator_input():
    from xopt.generators.remote.http import HTTPGenerator

    data = {"timeout": 12.0}
    assert HTTPGenerator._resolve_generator(data) is data


def test_http_generator_rejects_generator_without_name():
    import pytest
    from xopt.generators.remote.http import HTTPGenerator

    with pytest.raises(ValueError, match="must include a 'name' key"):
        HTTPGenerator._resolve_generator({"generator": {}})


def test_http_generator_yaml_inherits_top_level_vocs():
    from xopt import Xopt

    X = Xopt.from_yaml(
        """
        generator:
          name: http
          base_url: http://127.0.0.1:8002
          generator:
            name: upper_confidence_bound
            beta: 0.1
        evaluator:
          function: xopt.resources.test_functions.sinusoid_1d.evaluate_sinusoid
        vocs:
          variables: {x1: [0, 6.28]}
          objectives: {y1: MINIMIZE}
        """
    )

    assert isinstance(X.generator, HTTPGenerator)
    assert X.generator.generator.vocs.variable_names == ["x1"]


def test_http_generator_api_key_not_serialized():
    from xopt import Xopt
    from xopt.evaluator import Evaluator
    from xopt.resources.test_functions.sinusoid_1d import evaluate_sinusoid

    vocs = VOCS(variables={"x1": [0, 6.28]}, objectives={"y1": "MINIMIZE"})
    generator = HTTPGenerator(
        base_url="http://127.0.0.1:8002",
        generator=RandomGenerator(vocs=vocs),
        api_key="secret-token",
    )
    X = Xopt(generator=generator, evaluator=Evaluator(function=evaluate_sinusoid))

    assert "secret-token" not in X.yaml()
    assert "secret-token" not in generator.to_json()
    assert generator._http.headers["Authorization"] == "Bearer secret-token"


@pytest.mark.parametrize("wrapper", [HTTPGenerator, CloudpickleHTTPGenerator])
def test_remote_generator_xopt_dump_round_trip(wrapper):
    from xopt import Xopt
    from xopt.evaluator import Evaluator
    from xopt.generators.bayesian.upper_confidence_bound import (
        UpperConfidenceBoundGenerator,
    )
    from xopt.resources.test_functions.sinusoid_1d import evaluate_sinusoid

    vocs = VOCS(variables={"x1": [0, 6.28]}, objectives={"y1": "MINIMIZE"})
    X = Xopt(
        generator=wrapper(
            base_url="http://127.0.0.1:8002",
            generator=UpperConfidenceBoundGenerator(vocs=vocs, beta=0.3),
            api_key="secret-token",
        ),
        evaluator=Evaluator(function=evaluate_sinusoid),
    )

    reloaded = Xopt.from_yaml(X.yaml())

    assert type(reloaded.generator) is wrapper
    assert isinstance(reloaded.generator.generator, UpperConfidenceBoundGenerator)
    assert reloaded.generator.generator.beta == 0.3
    assert reloaded.generator.api_key is None


def _turbo_xopt(base_url, tmp_path, **wrapper_kwargs):
    from xopt import Xopt
    from xopt.evaluator import Evaluator
    from xopt.generators.bayesian.upper_confidence_bound import (
        UpperConfidenceBoundGenerator,
    )
    from xopt.resources.test_functions.sinusoid_1d import evaluate_sinusoid

    vocs = VOCS(variables={"x1": [0, 6.28]}, objectives={"y1": "MINIMIZE"})
    return Xopt(
        generator=HTTPGenerator(
            base_url=base_url,
            generator=UpperConfidenceBoundGenerator(
                vocs=vocs, turbo_controller="optimize"
            ),
            **wrapper_kwargs,
        ),
        evaluator=Evaluator(function=evaluate_sinusoid),
        xopt_dump_file=str(tmp_path / "dump.yaml"),
    )


def _dumped_turbo_center(tmp_path):
    import yaml

    config = yaml.safe_load((tmp_path / "dump.yaml").read_text())
    return config["generator"]["generator"]["turbo_controller"]["center_x"]


class TestHTTPGeneratorPullOnDump:
    def test_automatic_dump_matches_server_state(self, server_url, tmp_path):
        X = _turbo_xopt(server_url, tmp_path)
        X.random_evaluate(3)
        X.step()

        # TuRBO state only changes server-side; the dump must still include it
        center = _dumped_turbo_center(tmp_path)
        assert center is not None
        assert center == X.generator.generator.turbo_controller.center_x

        X.generator.finalize()

    def test_pull_on_dump_disabled_keeps_last_copy(self, server_url, tmp_path):
        X = _turbo_xopt(server_url, tmp_path, pull_on_dump=False)
        X.random_evaluate(3)
        X.step()

        assert _dumped_turbo_center(tmp_path) is None

        X.generator.finalize()

    def test_unreachable_server_warns_and_still_dumps(self, server_url, tmp_path):
        X = _turbo_xopt(server_url, tmp_path)
        X.random_evaluate(3)
        X.step()

        X.generator.base_url = "http://127.0.0.1:1"
        (tmp_path / "dump.yaml").unlink()
        with pytest.warns(UserWarning, match="could not reach"):
            X.dump()
        assert (tmp_path / "dump.yaml").exists()

        X.generator._session_id = None


def test_http_generator_unknown_private_attribute_raises():
    import pytest
    from xopt.generators.remote.http import HTTPGenerator

    generator = HTTPGenerator.model_construct()

    with pytest.raises(AttributeError, match="_missing_attribute"):
        getattr(generator, "_missing_attribute")
