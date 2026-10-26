"""Regression tests for cloudpickle payload type validation."""

import base64
from types import SimpleNamespace
from unittest.mock import patch

import cloudpickle
import pytest
from fastapi import HTTPException

from xopt.generators.remote import cloudpickle_server
from xopt.generators.remote.cloudpickle import CloudpickleHTTPGenerator


def test_cloudpickle_client_rejects_non_generator_response():
    """The client must reject an unexpected deserialized object."""
    payload = base64.b64encode(cloudpickle.dumps({"not": "a generator"})).decode()

    client = CloudpickleHTTPGenerator.model_construct()
    response = SimpleNamespace(json=lambda: {"payload": payload})

    with (
        patch.object(client, "_ensure_session", return_value="test-session"),
        patch.object(client, "_request", return_value=response),
        pytest.raises(TypeError, match="non-Generator object"),
    ):
        client.pull()


def test_cloudpickle_server_rejects_non_generator_payload():
    """The server must reject a valid pickle containing the wrong object type."""
    payload = base64.b64encode(cloudpickle.dumps({"not": "a generator"})).decode()

    with pytest.raises(HTTPException) as exc:
        cloudpickle_server.create_cloudpickle_session(
            cloudpickle_server.PicklePayload(payload=payload)
        )

    assert exc.value.status_code == 400
    assert "payload is not an Xopt Generator" in exc.value.detail
