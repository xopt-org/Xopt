"""Error and fallback paths of the remote generator client and server.

Run from repository root:
    python -m pytest xopt/tests/generators/remote/test_remote_edge_cases.py --no-cov
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gest_api.vocs import VOCS

from xopt.errors import XoptError
from xopt.generators.random import RandomGenerator
from xopt.generators.remote import server
from xopt.generators.remote.http import (
    HTTPGenerator,
    _error_detail,
    _records_json,
    _scalar,
)
from xopt.generators.remote.server import AUTH_TOKEN_ENV_VAR

TOKEN = "edge-case-token"
VOCS_2D = VOCS(
    variables={"x1": [0.0, 1.0], "x2": [0.0, 1.0]},
    objectives={"y1": "MINIMIZE"},
)


@pytest.fixture
def in_process(monkeypatch):
    """The server app in this process and a client wired to it."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv(AUTH_TOKEN_ENV_VAR, TOKEN)
    generator = HTTPGenerator(
        base_url="http://testserver",
        api_key=TOKEN,
        generator=RandomGenerator(vocs=VOCS_2D),
    )
    with TestClient(server.app) as client:
        generator._http = client
        yield server, generator
        generator.finalize()


# --- client: record encoding ---------------------------------------------------


def test_records_json_encodes_pandas_and_numpy_scalars():
    frame = pd.DataFrame(
        {
            "when": [pd.Timestamp("2026-01-02T03:04:05")],
            "missing_int": pd.array([pd.NA], dtype="Int64"),
            "missing_time": [pd.NaT],
            "value": np.array([1.5], dtype=np.float32),
            "flag": np.array([True]),
        }
    )
    (row,) = json.loads(_records_json("rows", frame))["rows"]
    assert row["when"] == "2026-01-02T03:04:05"
    assert row["missing_int"] is None
    assert row["missing_time"] is None
    assert row["value"] == pytest.approx(1.5)
    assert row["flag"] is True


def test_scalar_unwraps_numpy_scalars():
    # pandas 3 converts these itself in to_dict; older versions hand them to json
    assert _scalar(np.float64(1.5)) == 1.5
    assert _scalar(np.int64(3)) == 3
    assert type(_scalar(np.int64(3))) is int


def test_records_json_rejects_unknown_objects():
    frame = pd.DataFrame({"obj": [object()]})
    with pytest.raises(TypeError, match="not JSON serializable"):
        _records_json("rows", frame)


def test_records_json_of_none_is_empty():
    assert json.loads(_records_json("data", None)) == {"data": []}


# --- client: error detail extraction -------------------------------------------


def _response(text, reason="Bad Request"):
    def parse():
        return json.loads(text)

    return SimpleNamespace(text=text, reason=reason, json=parse)


def test_error_detail_prefers_fastapi_detail():
    assert _error_detail(_response('{"detail": "unknown session"}')) == (
        "unknown session"
    )


def test_error_detail_falls_back_to_body_then_reason():
    assert _error_detail(_response("Internal Server Error")) == "Internal Server Error"
    assert _error_detail(_response('{"other": 1}')) == '{"other": 1}'
    assert _error_detail(_response("   ", reason="Gateway Timeout")) == (
        "Gateway Timeout"
    )
    # a JSON body that is not an object has no `.get`
    assert _error_detail(_response("[1, 2]")) == "[1, 2]"


def test_error_detail_is_truncated():
    assert len(_error_detail(_response(json.dumps({"detail": "x" * 2000})))) == 500


# --- client: data assignment and session rollback ------------------------------


def test_assigning_a_dict_as_data_is_pushed_as_a_frame(in_process):
    srv, generator = in_process
    generator.generate(1)
    generator.data = {"x1": [0.1], "x2": [0.2], "y1": [1.0]}
    remote = srv._sessions[generator._session_id].generator.data
    assert remote["y1"].tolist() == [1.0]
    assert isinstance(generator.data, pd.DataFrame)


def test_failed_push_and_failed_cleanup_still_drop_the_session(in_process):
    """When the server is unreachable after creating a session, both the data
    push and the cleanup DELETE fail. The client must still forget the session
    and surface the push error."""
    srv, generator = in_process
    generator.ingest([{"x1": 0.1, "x2": 0.2, "y1": 1.0}])
    generator.finalize()

    calls = []

    def failing_request(method, path, **kwargs):
        calls.append((method, path))
        if method == "POST" and path == "/sessions":
            return SimpleNamespace(json=lambda: {"session_id": "ghost"})
        raise XoptError(f"{method} {path} unreachable")

    generator._request = failing_request
    with pytest.raises(XoptError, match="PUT /sessions/ghost/data unreachable"):
        generator.generate(1)
    assert generator._session_id is None
    assert ("DELETE", "/sessions/ghost") in calls, "cleanup of the new session"


# --- server: error branches ----------------------------------------------------


class _Broken(RandomGenerator):
    """A generator whose operations fail with ordinary exceptions."""

    def generate(self, n_candidates):
        raise RuntimeError("cannot generate")

    def ingest(self, results):
        raise ValueError("cannot ingest")

    def __setattr__(self, name, value):
        if name == "data" and value is not None and len(value) > 0:
            raise ValueError("cannot store data")
        super().__setattr__(name, value)


def _register(srv, generator):
    session_id = "edge-case-session"
    with srv._lock:
        srv._sessions[session_id] = srv.GeneratorSession(generator)
    return session_id


def test_server_reports_generator_failures(in_process):
    srv, generator = in_process
    session_id = _register(srv, _Broken(vocs=VOCS_2D))
    try:
        generator._session_id = session_id
        with pytest.raises(XoptError, match="500.*cannot generate"):
            generator.generate(1)
        with pytest.raises(XoptError, match="400.*cannot ingest"):
            generator.ingest([{"x1": 0.1, "x2": 0.2, "y1": 1.0}])
        with pytest.raises(XoptError, match="400.*cannot store data"):
            generator.data = pd.DataFrame({"x1": [0.1], "x2": [0.2], "y1": [1.0]})
    finally:
        generator._session_id = None
        with srv._lock:
            srv._sessions.pop(session_id, None)


def test_server_reports_unknown_session_on_every_route(in_process):
    srv, generator = in_process
    generator._session_id = "no-such-session"
    try:
        with pytest.raises(XoptError, match="404.*unknown session"):
            generator.ingest([{"x1": 0.1, "x2": 0.2, "y1": 1.0}])
        with pytest.raises(XoptError, match="404.*unknown session"):
            generator.data = pd.DataFrame({"x1": [0.1], "x2": [0.2], "y1": [1.0]})
        with pytest.raises(XoptError, match="404.*unknown session"):
            generator.pull()
    finally:
        generator._session_id = None


def test_main_reads_host_and_port_from_the_environment(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "uvicorn.run", lambda app, host, port: calls.append((app, host, port))
    )
    monkeypatch.setenv("XOPT_HTTP_GENERATOR_HOST", "0.0.0.0")
    monkeypatch.setenv("XOPT_HTTP_GENERATOR_PORT", "9001")
    server.main()
    assert calls == [(server.app, "0.0.0.0", 9001)]
