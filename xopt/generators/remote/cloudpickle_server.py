"""Opt-in cloudpickle endpoints for the upstream Xopt HTTP session server.

The existing upstream routes remain responsible for suggestions, ingestion,
state access, and session deletion. These two additional routes support custom
Python classes defined inside a scientist's notebook, where ordinary pickle
cannot resolve the class by an importable module name.

SECURITY: cloudpickle.loads() can execute arbitrary code. This prototype is
ONLY suitable for trusted, isolated clients; do not expose it publicly.
"""

import base64
import uuid

import cloudpickle
from fastapi import HTTPException
from pydantic import BaseModel

from xopt.generator import Generator
from xopt.generators.remote import server as upstream

# Extend the *same* FastAPI app so standard HTTPGenerator clients continue to
# use the unchanged upstream API. Import this module when starting Uvicorn.
app = upstream.app


class PicklePayload(BaseModel):
    """Base64-encoded cloudpickle bytes carried in a JSON request/response."""

    payload: str


@app.post("/sessions/cloudpickle", response_model=upstream.SessionResponse)
def create_cloudpickle_session(request: PicklePayload):
    """Deserialize a trusted generator and register it as an upstream session.

    Unlike standard module-name-based serialization, cloudpickle can include
    definitions of notebook-local classes. Installed third-party dependencies
    (e.g. torch) must still exist on the server.
    """
    try:
        # Base64 makes binary pickle bytes safe to transport in JSON. It is
        # encoding, NOT encryption or validation of the embedded Python code.
        obj = cloudpickle.loads(base64.b64decode(request.payload, validate=True))
        if not isinstance(obj, Generator):
            raise ValueError("payload is not an Xopt Generator")
    except Exception as exc:
        # Note: this check occurs AFTER deserialization; it is not a security
        # sandbox and cannot prevent execution of a malicious pickle payload.
        raise HTTPException(status_code=400, detail=f"invalid generator payload: {exc}") from exc

    session_id = str(uuid.uuid4())
    # Integration tradeoff: reuse upstream's private session registry and lock
    # to preserve its lifecycle and other endpoints. These private attributes
    # may change in future upstream releases and should be revisited before
    # merging this functionality into Xopt proper.
    with upstream._lock:
        upstream._sessions[session_id] = obj
    return upstream.SessionResponse(session_id=session_id)


@app.get("/sessions/{session_id}/state/cloudpickle")
def get_cloudpickle_state(session_id: str):
    """Return the current generator, including notebook-local class definitions.

    Session lookup and unknown-session handling are delegated to upstream.
    The client uses this response to refresh its local generator mirror.
    """
    obj = upstream._get_session(session_id)
    return {"payload": base64.b64encode(cloudpickle.dumps(obj)).decode("ascii")}
