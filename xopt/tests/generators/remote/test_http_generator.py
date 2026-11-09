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


SERVER_TOKEN = "module-test-token"


@pytest.fixture(scope="module")
def server_url():
    """Normal upstream HTTP server, one independent process per test module."""
    with _uvicorn_subprocess(auth_token=SERVER_TOKEN) as url:
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


@pytest.fixture
def in_process_server(monkeypatch):
    """The JSON server app in this process, so a test can read the server's
    session state. Yields (server module, HTTPGenerator wired to it)."""
    from fastapi.testclient import TestClient

    from xopt.generators.remote import server

    monkeypatch.setenv(AUTH_TOKEN_ENV_VAR, SERVER_TOKEN)
    generator = HTTPGenerator(
        base_url="http://testserver",
        api_key=SERVER_TOKEN,
        generator=RandomGenerator(vocs=RANDOM_VOCS),
    )
    with TestClient(server.app) as client:
        generator._http = client
        yield server, generator
        generator.finalize()


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


@pytest.mark.parametrize("token", [None, "", "   "])
def test_server_refuses_startup_without_token(token):
    with pytest.raises(RuntimeError, match=AUTH_TOKEN_ENV_VAR):
        with _uvicorn_subprocess(auth_token=token):
            pass


def test_server_rejects_missing_or_wrong_token(server_url):
    body = {"generator_name": "random", "state": {}}
    assert requests.post(server_url + "/sessions", json=body).status_code == 401
    wrong = {"Authorization": "Bearer wrong-token"}
    assert (
        requests.post(server_url + "/sessions", json=body, headers=wrong).status_code
        == 401
    )


def test_server_rejects_requests_if_token_missing_at_request_time(monkeypatch):
    from fastapi import HTTPException

    from xopt.generators.remote.server import _require_auth

    monkeypatch.delenv(AUTH_TOKEN_ENV_VAR, raising=False)
    with pytest.raises(HTTPException) as exc:
        _require_auth(authorization=None)
    assert exc.value.status_code == 503


class TestHTTPGeneratorRandom:
    def test_suggest_within_bounds(self, server_url):
        generator = HTTPGenerator(
            base_url=server_url,
            api_key=SERVER_TOKEN,
            generator=RandomGenerator(vocs=RANDOM_VOCS),
        )
        points = generator.generate(5)
        assert len(points) == 5
        for point in points:
            assert 0.0 <= point["x1"] <= 1.0
            assert 0.0 <= point["x2"] <= 1.0
        generator.finalize()

    def test_ingest_updates_local_cache(self, server_url):
        generator = HTTPGenerator(
            base_url=server_url,
            api_key=SERVER_TOKEN,
            generator=RandomGenerator(vocs=RANDOM_VOCS),
        )
        results = pd.DataFrame({"x1": [0.1, 0.2], "x2": [0.3, 0.4], "y1": [1.0, 2.0]})
        generator.ingest(results.to_dict(orient="records"))
        assert generator.data is not None
        assert len(generator.data) == 2
        generator.finalize()

    def test_full_replace_updates_remote_mirror(self, server_url):
        generator = HTTPGenerator(
            base_url=server_url,
            api_key=SERVER_TOKEN,
            generator=RandomGenerator(vocs=RANDOM_VOCS),
        )
        # create a session first
        generator.generate(1)

        new_data = pd.DataFrame({"x1": [0.5], "x2": [0.5], "y1": [3.0]})
        generator.data = new_data

        mirror = generator.pull()
        assert len(mirror.data) == 1
        assert mirror.data.iloc[0]["x1"] == pytest.approx(0.5)
        generator.finalize()

    def test_failed_replace_leaves_local_data_unchanged(self, server_url):
        generator = HTTPGenerator(
            base_url=server_url,
            api_key=SERVER_TOKEN,
            generator=RandomGenerator(vocs=RANDOM_VOCS),
        )
        generator.ingest([{"x1": 0.1, "x2": 0.2, "y1": 1.0}])
        before = generator.data.copy()

        generator.base_url = "http://127.0.0.1:1"
        with pytest.raises(XoptError):
            generator.data = pd.DataFrame({"x1": [0.5], "x2": [0.5], "y1": [3.0]})
        # client and server must still agree: the replacement never happened
        pd.testing.assert_frame_equal(generator.data, before)

        generator.base_url = server_url
        generator.finalize()

    def test_data_reaches_server_exactly(self, in_process_server):
        """Floats, NaN, inf, numpy ints, bools, strings and array cells survive
        the trip. `DataFrame.to_json` rounds to 10 decimal places, which turned
        small values such as an emittance of 1e-9 into 0 on the server."""
        server, generator = in_process_server
        sent = pd.DataFrame(
            {
                "x1": [1.2345678901234567e-12, 0.12345678901234567, -0.0],
                "x2": [5e-324, 0.2, 0.3],
                "y1": [np.inf, 7.0, np.nan],
                "xopt_error": [False, False, True],
                "xopt_error_str": [None, None, "boom"],
                "n": np.array([1, 2, 3], dtype=np.int64),
                "arr": [np.array([1.0, 2.0]), np.array([3.0]), np.array([])],
            }
        )

        def remote():
            # the server's own frame, not the client's mirror of it
            return server._sessions[generator._session_id].generator.data

        def check(remote_data, expected):
            for col in ("x1", "x2", "y1"):
                assert np.array_equal(
                    remote_data[col].to_numpy(dtype=float),
                    expected[col].to_numpy(dtype=float),
                    equal_nan=True,
                ), col
            assert np.signbit(remote_data["x1"].iloc[-1]) == np.signbit(
                expected["x1"].iloc[-1]
            )
            assert remote_data["xopt_error"].tolist() == expected["xopt_error"].tolist()
            # None arrives as NaN on both sides; compare the strings only
            assert [v for v in remote_data["xopt_error_str"] if isinstance(v, str)] == [
                v for v in expected["xopt_error_str"] if isinstance(v, str)
            ]
            assert remote_data["n"].tolist() == expected["n"].tolist()
            assert [list(a) for a in remote_data["arr"]] == [
                list(a) for a in expected["arr"]
            ]

        # ingest appends, assigning data replaces: check both paths
        generator.add_data(sent.iloc[:2])
        check(remote(), sent.iloc[:2])
        generator.data = sent
        check(remote(), sent)

    def test_new_session_receives_existing_data(self, in_process_server):
        """A session created after finalize(), or for a deepcopy, starts from
        the data the client already holds."""
        import copy

        server, generator = in_process_server
        generator.ingest([{"x1": 0.1, "x2": 0.2, "y1": 1.0}])
        generator.finalize()

        generator.ingest([{"x1": 0.3, "x2": 0.4, "y1": 2.0}])
        remote = server._sessions[generator._session_id].generator.data
        assert len(remote) == 2
        assert remote["y1"].tolist() == [1.0, 2.0]

        copied = copy.deepcopy(generator)
        copied._http = generator._http
        copied.ingest([{"x1": 0.5, "x2": 0.6, "y1": 3.0}])
        assert copied._session_id != generator._session_id
        remote = server._sessions[copied._session_id].generator.data
        assert remote["y1"].tolist() == [1.0, 2.0, 3.0]
        copied.finalize()

    def test_failed_initial_push_drops_the_session(self, in_process_server):
        """If the client's data can't be sent to a new session, the session is
        not kept: otherwise later calls would run against a server missing it."""
        server, generator = in_process_server
        generator.ingest([{"x1": 0.1, "x2": 0.2, "y1": 1.0}])
        generator.finalize()

        real_push = generator._push_full_data
        calls = []

        def failing_push(data):
            calls.append(data)
            raise XoptError("simulated network failure")

        generator._push_full_data = failing_push
        with pytest.raises(XoptError, match="simulated"):
            generator.generate(1)
        assert generator._session_id is None
        assert len(server._sessions) == 0, "orphaned session left on the server"

        generator._push_full_data = real_push
        generator.ingest([{"x1": 0.3, "x2": 0.4, "y1": 2.0}])
        remote = server._sessions[generator._session_id].generator.data
        assert remote["y1"].tolist() == [1.0, 2.0]

    def test_server_refuses_remote_generator_names(self, server_url):
        """A wrapper running on the server would make it call arbitrary hosts."""
        inner = HTTPGenerator(
            base_url="http://127.0.0.1:1",
            generator=RandomGenerator(vocs=RANDOM_VOCS),
        )
        outer = HTTPGenerator(
            base_url=server_url, api_key=SERVER_TOKEN, generator=inner
        )
        with pytest.raises(XoptError, match="'http' cannot run on the server"):
            outer.generate(1)

    def test_api_key_assigned_after_construction_is_used(self, server_url):
        generator = HTTPGenerator(
            base_url=server_url,
            generator=RandomGenerator(vocs=RANDOM_VOCS),
        )
        with pytest.raises(XoptError, match="401"):
            generator.generate(1)
        generator.api_key = SERVER_TOKEN
        assert len(generator.generate(1)) == 1
        generator.finalize()

    def test_unknown_generator_name_raises(self, server_url):
        generator = HTTPGenerator(
            base_url=server_url,
            api_key=SERVER_TOKEN,
            generator=_UnregisteredGenerator(vocs=RANDOM_VOCS),
        )
        # the server's reason must reach the user, not just "400 Client Error"
        with pytest.raises(XoptError, match="unknown generator 'not_a_real_generator'"):
            generator.generate(1)

    def test_unknown_session_raises(self, server_url):
        generator = HTTPGenerator(
            base_url=server_url,
            api_key=SERVER_TOKEN,
            generator=RandomGenerator(vocs=RANDOM_VOCS),
        )
        generator.generate(1)
        # force a bogus session id to simulate an expired/unknown session
        generator._session_id = "does-not-exist"
        with pytest.raises(XoptError, match="unknown session 'does-not-exist'"):
            generator.generate(1)

    def test_auth_required(self, authenticated_server_url):
        server_url = authenticated_server_url

        unauthenticated = HTTPGenerator(
            base_url=server_url, generator=RandomGenerator(vocs=RANDOM_VOCS)
        )
        with pytest.raises(XoptError, match="401.*invalid or missing bearer token"):
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
            api_key=SERVER_TOKEN,
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
            api_key=SERVER_TOKEN,
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
            api_key=SERVER_TOKEN,
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

        generator = HTTPGenerator(
            base_url=server_url, api_key=SERVER_TOKEN, generator=bax_generator
        )

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

        generator = HTTPGenerator(
            base_url=server_url, api_key=SERVER_TOKEN, generator=bo_generator
        )

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
    # a notebook displaying the wrapper, or a logged exception, must not show it
    assert "secret-token" not in repr(generator)
    assert "secret-token" not in repr(X)


def test_http_generator_deepcopy_does_not_share_session():
    """Xopt deepcopies its generator; the copy must get its own session, or
    finalize() on one would close the other's."""
    import copy

    generator = HTTPGenerator(
        base_url="http://127.0.0.1:1",
        generator=RandomGenerator(vocs=RANDOM_VOCS),
        api_key="secret-token",
    )
    generator._session_id = "live-session"

    copied = copy.deepcopy(generator)
    assert copied._session_id is None
    assert copied._http is not generator._http
    assert copied.api_key == "secret-token"
    assert generator._session_id == "live-session"


def test_http_generator_rejects_mismatched_vocs():
    other = VOCS(variables={"x1": [0.0, 2.0]}, objectives={"y1": "MINIMIZE"})
    with pytest.raises(ValueError, match="vocs.*differ"):
        HTTPGenerator(
            base_url="http://127.0.0.1:1",
            generator=RandomGenerator(vocs=RANDOM_VOCS),
            vocs=other,
        )


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
        X = _turbo_xopt(server_url, tmp_path, api_key=SERVER_TOKEN)
        X.random_evaluate(3)
        X.step()

        # TuRBO state only changes server-side; the dump must still include it
        center = _dumped_turbo_center(tmp_path)
        assert center is not None
        assert center == X.generator.generator.turbo_controller.center_x

        X.generator.finalize()

    def test_pull_on_dump_disabled_keeps_last_copy(self, server_url, tmp_path):
        X = _turbo_xopt(server_url, tmp_path, api_key=SERVER_TOKEN, pull_on_dump=False)
        X.random_evaluate(3)
        X.step()

        assert _dumped_turbo_center(tmp_path) is None

        X.generator.finalize()

    def test_unreachable_server_warns_and_still_dumps(self, server_url, tmp_path):
        X = _turbo_xopt(server_url, tmp_path, api_key=SERVER_TOKEN)
        X.random_evaluate(3)
        X.step()

        X.generator.base_url = "http://127.0.0.1:1"
        (tmp_path / "dump.yaml").unlink()
        with pytest.warns(UserWarning, match="could not pull"):
            X.dump()
        assert (tmp_path / "dump.yaml").exists()

        X.generator._session_id = None

    def test_checkpoint_restores_into_new_session_and_continues(
        self, server_url, tmp_path
    ):
        """Full recovery: dump, lose the server session, reload the dump into a
        new session and keep optimizing with the TuRBO state intact."""
        import yaml

        from xopt import Xopt

        X = _turbo_xopt(server_url, tmp_path, api_key=SERVER_TOKEN)
        X.random_evaluate(3)
        X.step()
        X.step()
        n_before = len(X.data)
        old_session = X.generator._session_id

        config = yaml.safe_load((tmp_path / "dump.yaml").read_text())
        dumped_turbo = config["generator"]["generator"]["turbo_controller"]
        assert dumped_turbo["center_x"] is not None

        # the original session is gone for good
        X.generator.finalize()
        X.generator._session_id = old_session
        with pytest.raises(XoptError, match="unknown session"):
            X.generator.generate(1)
        X.generator._session_id = None

        # loading contacts the server, so the token has to be in place first
        with pytest.raises(XoptError, match="401"):
            Xopt.from_dict(yaml.safe_load((tmp_path / "dump.yaml").read_text()))
        config["generator"]["api_key"] = SERVER_TOKEN
        X2 = Xopt.from_dict(config)

        assert len(X2.data) == n_before
        assert X2.generator._session_id not in (None, old_session)
        turbo_keys = ("center_x", "length", "best_value")
        local_turbo = X2.generator.generator.turbo_controller
        remote_turbo = X2.generator.pull().turbo_controller
        for key in turbo_keys:
            assert getattr(local_turbo, key) == dumped_turbo[key]
            assert getattr(remote_turbo, key) == dumped_turbo[key]

        X2.step()
        assert len(X2.data) == n_before + 1
        assert _dumped_turbo_center(tmp_path) is not None

        X2.generator.finalize()


class _FakeResponse:
    def __init__(self, text):
        self.text = text


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the CPU-only client path")
def test_pull_maps_cuda_model_to_cpu(monkeypatch):
    """A model fitted on a GPU server must load on a CPU-only client."""
    from xopt.generators.bayesian.upper_confidence_bound import (
        UpperConfidenceBoundGenerator,
    )
    from xopt.generators.remote import server

    remote = UpperConfidenceBoundGenerator(vocs=BO_VOCS)
    remote.add_data(pd.DataFrame({"x": [0.1, 0.5, 0.9], "y": [1.0, 0.2, 0.7]}))
    remote.train_model()

    # the server serializes the state with the storages tagged as CUDA
    session_id = "cuda-state"
    with server._lock:
        server._sessions[session_id] = server.GeneratorSession(remote)
    monkeypatch.setattr(torch.serialization, "location_tag", lambda storage: "cuda:0")
    try:
        state_json = server.get_state(session_id).body.decode()
    finally:
        monkeypatch.undo()
        with server._lock:
            server._sessions.pop(session_id, None)
    assert "base64:" in state_json

    client = HTTPGenerator(
        base_url="http://127.0.0.1:1",
        generator=UpperConfidenceBoundGenerator(vocs=BO_VOCS),
    )
    client._session_id = "cuda-state"
    monkeypatch.setattr(
        client, "_request", lambda *args, **kwargs: _FakeResponse(state_json)
    )

    mirror = client.pull()
    assert mirror.model is not None
    assert all(p.device.type == "cpu" for p in mirror.model.parameters())
    assert all(b.device.type == "cpu" for b in mirror.model.buffers())


def test_pull_reconstruction_failure_is_xopt_error(monkeypatch):
    """Any failure while rebuilding the pulled state is an XoptError, so
    `sync_for_dump()` falls back to the last copy instead of breaking the dump."""
    client = HTTPGenerator(
        base_url="http://127.0.0.1:1",
        generator=RandomGenerator(vocs=RANDOM_VOCS),
    )
    client._session_id = "whatever"
    monkeypatch.setattr(
        client, "_request", lambda *args, **kwargs: _FakeResponse('{"vocs": 1}')
    )

    with pytest.raises(XoptError, match="failed to rebuild"):
        client.pull()
    with pytest.warns(UserWarning, match="could not pull"):
        client.sync_for_dump()


def test_http_generator_unknown_private_attribute_raises():
    import pytest
    from xopt.generators.remote.http import HTTPGenerator

    generator = HTTPGenerator.model_construct()

    with pytest.raises(AttributeError, match="_missing_attribute"):
        getattr(generator, "_missing_attribute")
