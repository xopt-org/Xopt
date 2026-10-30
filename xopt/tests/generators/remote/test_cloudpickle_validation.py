"""Regression tests for cloudpickle payload type validation."""

import base64
from types import SimpleNamespace
from unittest.mock import patch

import cloudpickle
import pytest
from fastapi import HTTPException

from xopt.errors import XoptError
from xopt.generators.remote import cloudpickle_server
from xopt.generators.remote.cloudpickle import CloudpickleHTTPGenerator


def test_cloudpickle_client_rejects_non_generator_response():
    """The client must reject an unexpected deserialized object, as an XoptError
    so that sync_for_dump() can fall back to the last pulled copy."""
    payload = base64.b64encode(cloudpickle.dumps({"not": "a generator"})).decode()

    client = CloudpickleHTTPGenerator.model_construct()
    response = SimpleNamespace(json=lambda: {"payload": payload})

    with (
        patch.object(client, "_ensure_session", return_value="test-session"),
        patch.object(client, "_request", return_value=response),
        pytest.raises(XoptError, match="non-Generator object"),
    ):
        client.pull()


def test_cloudpickle_routes_stay_off_the_plain_server_app():
    """Importing the cloudpickle server must not add its unpickling endpoints
    to `xopt.generators.remote.server:app`."""
    from xopt.generators.remote import server

    plain = set(server.app.openapi()["paths"])
    extended = set(cloudpickle_server.app.openapi()["paths"])

    assert "/sessions/cloudpickle" not in plain
    assert "/sessions/cloudpickle" in extended
    assert "/sessions/{session_id}/state/cloudpickle" in extended
    # the standard routes are served by both
    assert "/sessions/{session_id}/suggest" in plain & extended


def test_cloudpickle_server_rejects_non_generator_payload():
    """The server must reject a valid pickle containing the wrong object type."""
    payload = base64.b64encode(cloudpickle.dumps({"not": "a generator"})).decode()

    with pytest.raises(HTTPException) as exc:
        cloudpickle_server.create_cloudpickle_session(
            cloudpickle_server.PicklePayload(payload=payload)
        )

    assert exc.value.status_code == 400
    assert "payload is not an Xopt Generator" in exc.value.detail
