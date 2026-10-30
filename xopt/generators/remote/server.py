"""Reference HTTP server exposing a single generator per session.

Run directly with `python -m xopt.generators.remote.server`, or mount `app`
behind your own uvicorn/gunicorn setup. This is a minimal, single-process
reference implementation: sessions live in memory and are lost on restart.
"""

import logging
import os
import secrets
import threading
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Iterator
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from xopt.generator import Generator
from xopt.generators import all_generator_names, get_generator_dynamic

logger = logging.getLogger(__name__)

# bearer token required of all callers. The server refuses to start without
# it: session creation unpickles torch modules from the client's config, so an
# open server would let anyone who can reach the port run code on this host.
AUTH_TOKEN_ENV_VAR = "XOPT_HTTP_GENERATOR_TOKEN"


@dataclass
class GeneratorSession:
    generator: Generator
    lock: threading.Lock = field(default_factory=threading.Lock)
    closed: bool = False


_sessions: dict[str, GeneratorSession] = {}
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


def _configured_token() -> Optional[str]:
    token = os.environ.get(AUTH_TOKEN_ENV_VAR)
    if token is None or not token.strip():
        return None
    return token


def _require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    token = _configured_token()
    # fail closed if the app is served without the startup check (e.g. mounted
    # inside another app, whose lifespan does not run this one's)
    if token is None:
        raise HTTPException(
            status_code=503, detail=f"server has no {AUTH_TOKEN_ENV_VAR} configured"
        )
    # compare bytes: compare_digest rejects non-ASCII str, which would be a 500
    expected = f"Bearer {token}".encode()
    if not secrets.compare_digest((authorization or "").encode(), expected):
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


@asynccontextmanager
async def _lifespan(application: FastAPI):
    if _configured_token() is None:
        raise RuntimeError(
            f"HTTPGenerator server requires a nonempty {AUTH_TOKEN_ENV_VAR}"
        )
    yield


# the session routes live on a router so other apps (the cloudpickle server) can
# include them without sharing, and mutating, this module's `app`
router = APIRouter()

app = FastAPI(
    title="Xopt HTTPGenerator reference server",
    dependencies=[Depends(_require_auth)],
    lifespan=_lifespan,
)


def _get_session(session_id: str) -> GeneratorSession:
    with _lock:
        session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown session {session_id!r}")
    return session


@contextmanager
def _locked_session(session_id: str) -> Iterator[Generator]:
    """Protect one generator operation, including against concurrent deletion."""
    session = _get_session(session_id)
    with session.lock:
        if session.closed:
            raise HTTPException(
                status_code=404, detail=f"unknown session {session_id!r}"
            )
        yield session.generator


@router.post("/sessions", response_model=SessionResponse)
def create_session(request: CreateSessionRequest) -> SessionResponse:
    # a remote wrapper running here would make this server send requests, with
    # a caller-chosen URL and token, to hosts only it can reach
    if request.generator_name in all_generator_names["remote"]:
        raise HTTPException(
            status_code=400,
            detail=f"generator {request.generator_name!r} cannot run on the server",
        )
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
        _sessions[session_id] = GeneratorSession(generator)
    return SessionResponse(session_id=session_id)


@router.post("/sessions/{session_id}/suggest", response_model=SuggestResponse)
def suggest(session_id: str, request: SuggestRequest) -> SuggestResponse:
    try:
        with _locked_session(session_id) as generator:
            points = generator.suggest(request.num_points)
    except HTTPException:
        # e.g. unknown session; don't turn it into a generic 500
        raise
    except Exception as exc:
        logger.exception("generator.suggest failed")
        raise HTTPException(
            status_code=500, detail=f"generator failed to produce points: {exc}"
        )
    return SuggestResponse(points=points)


@router.post("/sessions/{session_id}/ingest", status_code=204)
def ingest(session_id: str, request: IngestRequest) -> None:
    try:
        with _locked_session(session_id) as generator:
            generator.ingest(request.results)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("generator.ingest failed")
        raise HTTPException(status_code=400, detail=f"failed to ingest results: {exc}")


@router.put("/sessions/{session_id}/data", status_code=204)
def replace_data(session_id: str, request: ReplaceDataRequest) -> None:
    try:
        with _locked_session(session_id) as generator:
            generator.data = pd.DataFrame(request.data)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("replacing generator data failed")
        raise HTTPException(status_code=400, detail=f"failed to replace data: {exc}")


@router.get("/sessions/{session_id}/state")
def get_state(session_id: str) -> Response:
    with _locked_session(session_id) as generator:
        state_json = generator.to_json(serialize_torch=True, serialize_inline=True)
    return Response(content=state_json, media_type="application/json")


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str) -> None:
    with _lock:
        session = _sessions.pop(session_id, None)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown session {session_id!r}")
    with session.lock:
        session.closed = True
        finalize = getattr(session.generator, "finalize", None)
        if callable(finalize):
            try:
                finalize()
            except Exception:
                logger.exception("generator.finalize failed")


app.include_router(router)


def main() -> None:
    import uvicorn

    host = os.environ.get("XOPT_HTTP_GENERATOR_HOST", "127.0.0.1")
    port = int(os.environ.get("XOPT_HTTP_GENERATOR_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    main()
