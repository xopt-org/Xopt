"""Opt-in cloudpickle endpoints for the upstream Xopt HTTP session server.

The existing upstream routes remain responsible for suggestions, ingestion,
state access, and session deletion. These two additional routes support custom
Python classes defined inside a scientist's notebook, where ordinary pickle
cannot resolve the class by an importable module name.

SECURITY: cloudpickle.loads() can execute arbitrary code. This prototype is
ONLY suitable for trusted, isolated clients; do not expose it publicly.
"""

import base64
import io
import pickle
import uuid
from contextlib import asynccontextmanager

import cloudpickle
import torch
from types import SimpleNamespace
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from xopt.generator import Generator
from xopt.generators.remote import server as upstream


def _require_cloudpickle_token_on_startup() -> None:
    """Fail closed before accepting requests capable of unpickling code."""
    import os

    token = os.environ.get(upstream.AUTH_TOKEN_ENV_VAR)
    if not token or not token.strip():
        raise RuntimeError(
            f"Cloudpickle server requires a nonempty {upstream.AUTH_TOKEN_ENV_VAR}"
        )


@asynccontextmanager
async def _cloudpickle_lifespan(application):
    _require_cloudpickle_token_on_startup()
    async with upstream._lifespan(application):
        yield


# A separate app that includes the upstream routes, so importing this module
# does not add the unpickling endpoints to `xopt.generators.remote.server:app`.
# Standard HTTPGenerator clients work against it unchanged.
app = FastAPI(
    title="Xopt HTTPGenerator reference server (with cloudpickle)",
    dependencies=[Depends(upstream._require_auth)],
    lifespan=_cloudpickle_lifespan,
)
app.include_router(upstream.router)


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
        raise HTTPException(
            status_code=400, detail=f"invalid generator payload: {exc}"
        ) from exc

    session_id = str(uuid.uuid4())
    # Integration tradeoff: reuse upstream's private session registry and lock
    # to preserve its lifecycle and other endpoints. These private attributes
    # may change in future upstream releases and should be revisited before
    # merging this functionality into Xopt proper.
    with upstream._lock:
        upstream._sessions[session_id] = upstream.GeneratorSession(obj)
    return upstream.SessionResponse(session_id=session_id)


@app.get("/sessions/{session_id}/state/cloudpickle")
def get_cloudpickle_state(session_id: str):
    """Return the current generator, including notebook-local class definitions.

    Session lookup and unknown-session handling are delegated to upstream.
    The client uses this response to refresh its local generator mirror.
    """
    with upstream._locked_session(session_id) as obj:
        # Serialize the live generator without modifying its CUDA tensors.
        # A separate CPU-safe snapshot is needed for CPU-only HTTP clients.
        #
        # PyTorch's map_location applies to tensor storage throughout the
        # serialized object, including fitted GP models and BAX results.
        # Cloudpickle supplies the Pickler; standard pickle supplies the
        # Unpickler required by torch.load().
        cloudpickle_adapter = SimpleNamespace(
            Pickler=cloudpickle.CloudPickler,
            Unpickler=pickle.Unpickler,
            HIGHEST_PROTOCOL=pickle.HIGHEST_PROTOCOL,
            __name__="cloudpickle_adapter",
        )

        buffer = io.BytesIO()
        torch.save(obj, buffer, pickle_module=cloudpickle_adapter)
        buffer.seek(0)

        snapshot = torch.load(
            buffer,
            map_location="cpu",
            weights_only=False,
            pickle_module=cloudpickle_adapter,
        )

        # The snapshot is independent of the live GPU generator.
        payload = base64.b64encode(cloudpickle.dumps(snapshot)).decode("ascii")
        return {"payload": payload}
