"""Reference HTTP server exposing a single generator per session.

Run directly with `python -m xopt.generators.remote.server`, or mount `app`
behind your own uvicorn/gunicorn setup. This is a minimal, single-process
reference implementation: sessions live in memory and are lost on restart.
"""

import logging
import os
import threading
import uuid
from typing import Any, Optional

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from xopt.generator import Generator
from xopt.generators import get_generator_dynamic

logger = logging.getLogger(__name__)

# bearer token required of callers when set; the server is open if unset, which
# is only appropriate on a trusted/loopback network
AUTH_TOKEN_ENV_VAR = "XOPT_HTTP_GENERATOR_TOKEN"

_sessions: dict[str, Generator] = {}
_lock = threading.Lock()


class CreateSessionRequest(BaseModel):
    generator_name: str
    # full generator state (as produced by Generator.to_json), vocs included
    state: dict[str, Any]


class SessionResponse(BaseModel):
    session_id: str


class SuggestRequest(BaseModel):
    num_points: Optional[int] = None


class SuggestResponse(BaseModel):
    points: list[dict[str, Any]]


class IngestRequest(BaseModel):
    results: list[dict[str, Any]]


class ReplaceDataRequest(BaseModel):
    data: list[dict[str, Any]]


def _require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    token = os.environ.get(AUTH_TOKEN_ENV_VAR)
    if token is None:
        return
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


app = FastAPI(
    title="Xopt HTTPGenerator reference server",
    dependencies=[Depends(_require_auth)],
)


def _get_session(session_id: str) -> Generator:
    with _lock:
        generator = _sessions.get(session_id)
    if generator is None:
        raise HTTPException(status_code=404, detail=f"unknown session {session_id!r}")
    return generator


@app.post("/sessions", response_model=SessionResponse)
def create_session(request: CreateSessionRequest) -> SessionResponse:
    try:
        # restrict construction to the existing registry, never an arbitrary import path
        generator_class = get_generator_dynamic(request.generator_name)
    except KeyError:
        raise HTTPException(
            status_code=400, detail=f"unknown generator {request.generator_name!r}"
        )

    try:
        generator = generator_class.model_validate(request.state)
    except Exception as exc:
        logger.exception("failed to construct remote generator")
        raise HTTPException(
            status_code=400, detail=f"failed to construct generator: {exc}"
        )

    session_id = str(uuid.uuid4())
    with _lock:
        _sessions[session_id] = generator
    return SessionResponse(session_id=session_id)


@app.post("/sessions/{session_id}/suggest", response_model=SuggestResponse)
def suggest(session_id: str, request: SuggestRequest) -> SuggestResponse:
    generator = _get_session(session_id)
    try:
        points = generator.suggest(request.num_points)
    except Exception:
        logger.exception("generator.suggest failed")
        raise HTTPException(
            status_code=500, detail="generator failed to produce points"
        )
    return SuggestResponse(points=points)


@app.post("/sessions/{session_id}/ingest", status_code=204)
def ingest(session_id: str, request: IngestRequest) -> None:
    generator = _get_session(session_id)
    try:
        generator.ingest(request.results)
    except Exception as exc:
        logger.exception("generator.ingest failed")
        raise HTTPException(status_code=400, detail=f"failed to ingest results: {exc}")


@app.put("/sessions/{session_id}/data", status_code=204)
def replace_data(session_id: str, request: ReplaceDataRequest) -> None:
    generator = _get_session(session_id)
    generator.data = pd.DataFrame(request.data)


@app.get("/sessions/{session_id}/state")
def get_state(session_id: str) -> Response:
    generator = _get_session(session_id)
    state_json = generator.to_json(serialize_torch=True, serialize_inline=True)
    return Response(content=state_json, media_type="application/json")


@app.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str) -> None:
    with _lock:
        generator = _sessions.pop(session_id, None)
    if generator is None:
        raise HTTPException(status_code=404, detail=f"unknown session {session_id!r}")
    finalize = getattr(generator, "finalize", None)
    if callable(finalize):
        try:
            finalize()
        except Exception:
            logger.exception("generator.finalize failed")


def main() -> None:
    import uvicorn

    host = os.environ.get("XOPT_HTTP_GENERATOR_HOST", "127.0.0.1")
    port = int(os.environ.get("XOPT_HTTP_GENERATOR_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
