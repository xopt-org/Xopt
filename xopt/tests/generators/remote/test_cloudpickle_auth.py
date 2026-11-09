"""Authentication regressions for the opt-in cloudpickle HTTP server."""

import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager

import pytest
import requests

TOKEN_ENV = "XOPT_HTTP_GENERATOR_TOKEN"
TEST_TOKEN = "test-only-cloudpickle-secret"


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def _server(token):
    port = _port()
    url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.pop(TOKEN_ENV, None)
    if token is not None:
        env[TOKEN_ENV] = token
    proc = subprocess.Popen(
        [
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
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        yield proc, url
    finally:
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        if proc.stderr:
            proc.stderr.close()


def _wait_ready(proc, url):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"server exited early: {proc.stderr.read()}")
        try:
            if requests.get(url + "/openapi.json", timeout=0.25).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(0.05)
    pytest.fail("server did not start")


@pytest.mark.parametrize("token", [None, "", "   "])
def test_cloudpickle_server_refuses_startup_without_token(token):
    with _server(token) as (proc, _url):
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pytest.fail("cloudpickle server stayed running without a valid token")
        assert proc.returncode != 0
        assert TOKEN_ENV in proc.stderr.read()


def test_cloudpickle_and_inherited_routes_require_token():
    with _server(TEST_TOKEN) as (proc, url):
        _wait_ready(proc, url)
        routes = (
            ("/sessions/cloudpickle", {"payload": "not-a-pickle"}),
            ("/sessions", {"generator_name": "random", "state": {}}),
        )
        for path, payload in routes:
            assert requests.post(url + path, json=payload, timeout=5).status_code == 401
            assert (
                requests.post(
                    url + path,
                    json=payload,
                    headers={"Authorization": "Bearer wrong-token"},
                    timeout=5,
                ).status_code
                == 401
            )
            # Valid authentication must reach the route's payload validation.
            # These deliberately invalid bodies should produce 400 or 422, not 401.
            response = requests.post(
                url + path,
                json=payload,
                headers={"Authorization": f"Bearer {TEST_TOKEN}"},
                timeout=5,
            )
            assert response.status_code in (400, 422), (
                path,
                response.status_code,
                response.text,
            )
