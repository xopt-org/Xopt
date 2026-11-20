"""Reference HTTP server exposing a single generator per session.

Run directly with `python -m xopt.generators.remote.server`, or mount `app`
behind your own uvicorn/gunicorn setup. This is a minimal, single-process
reference implementation: sessions live in memory and are lost on restart.
"""

import json
import logging
import os
import secrets
import threading
import time
import uuid
import warnings
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Iterator
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

import xopt
from xopt.errors import DataError
from xopt.generator import Generator
from xopt.generators import all_generator_names, get_generator_dynamic

logger = logging.getLogger(__name__)

# bearer token required of all callers. The server refuses to start without
# it: session creation unpickles torch modules from the client's config, so an
# open server would let anyone who can reach the port run code on this host.
AUTH_TOKEN_ENV_VAR = "XOPT_HTTP_GENERATOR_TOKEN"


# upper bound on live sessions, so sessions leaked by crashed clients cannot
# exhaust the server's memory for everyone else
MAX_SESSIONS_ENV_VAR = "XOPT_HTTP_MAX_SESSIONS"
DEFAULT_MAX_SESSIONS = 64


@dataclass
class GeneratorSession:
    generator: Generator
    lock: threading.Lock = field(default_factory=threading.Lock)
    closed: bool = False
    generator_name: str = ""
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)


_sessions: dict[str, GeneratorSession] = {}
_lock = threading.Lock()


class CreateSessionRequest(BaseModel):
    generator_name: str
    # full generator state (as produced by Generator.to_json), vocs included
    state: dict[str, Any]


class SessionResponse(BaseModel):
    session_id: str
    # warnings raised while constructing the generator, e.g. CUDA requested on
    # a CPU-only server; the client re-raises them so the user sees them
    warnings: list[str] = []


class ReplaceStateRequest(BaseModel):
    # generator state without the fitted model; the server retrains on the
    # next suggest from the data it already holds
    state: dict[str, Any]


class SessionInfo(BaseModel):
    session_id: str
    generator_name: str
    created_at: float
    last_used_at: float
    n_data: int


class SessionListResponse(BaseModel):
    sessions: list[SessionInfo]
    max_sessions: int


class HealthResponse(BaseModel):
    status: str
    xopt_version: str
    cuda_available: bool
    sessions: int
    max_sessions: int


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
router = APIRouter(dependencies=[Depends(_require_auth)])

app = FastAPI(title="Xopt HTTPGenerator reference server", lifespan=_lifespan)

# warnings.catch_warnings() is process-global, and sync endpoints run in a
# thread pool; without this, two concurrent constructions would record each
# other's warnings
_warnings_lock = threading.Lock()


def _max_sessions() -> int:
    raw = os.environ.get(MAX_SESSIONS_ENV_VAR)
    try:
        return int(raw) if raw else DEFAULT_MAX_SESSIONS
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; using %d",
            MAX_SESSIONS_ENV_VAR,
            raw,
            DEFAULT_MAX_SESSIONS,
        )
        return DEFAULT_MAX_SESSIONS


def _register_session(generator: Generator, generator_name: str) -> str:
    """Store a generator as a new session, within the session cap."""
    session_id = str(uuid.uuid4())
    with _lock:
        if len(_sessions) >= _max_sessions():
            raise HTTPException(
                status_code=503,
                detail=f"too many sessions ({len(_sessions)}); finalize unused "
                f"sessions or raise {MAX_SESSIONS_ENV_VAR} on the server",
            )
        _sessions[session_id] = GeneratorSession(
            generator, generator_name=generator_name
        )
    return session_id


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
        session.last_used_at = time.time()
        yield session.generator


def _generator_error(exc: Exception, status_code: int, what: str) -> HTTPException:
    """HTTPException for a generator failure. A DataError is tagged so the
    client can re-raise it as a DataError and Xopt(strict=False) can keep
    going, as it does with a local generator."""
    detail: Any = f"{what}: {exc}"
    if isinstance(exc, DataError):
        detail = {"message": detail, "type": "DataError"}
        status_code = 422
    return HTTPException(status_code=status_code, detail=detail)


def _build_generator(
    generator_name: str, state: dict[str, Any]
) -> tuple[Generator, list[str]]:
    """Construct a generator of the registered class from its state, returning
    the warnings raised on the way (e.g. CUDA requested on a CPU-only host)."""
    # a remote wrapper running here would make this server send requests, with
    # a caller-chosen URL and token, to hosts only it can reach
    if generator_name in all_generator_names["remote"]:
        raise HTTPException(
            status_code=400,
            detail=f"generator {generator_name!r} cannot run on the server",
        )
    try:
        # restrict construction to the existing registry, never an arbitrary import path
        generator_class = get_generator_dynamic(generator_name)
    except KeyError:
        raise HTTPException(
            status_code=400, detail=f"unknown generator {generator_name!r}"
        )

    try:
        with _warnings_lock, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            generator = generator_class.model_validate(state)
    except Exception as exc:
        logger.exception("failed to construct remote generator")
        raise HTTPException(
            status_code=400, detail=f"failed to construct generator: {exc}"
        )
    messages = [str(w.message) for w in caught]
    # BO generators only warn about this when they first train, which the
    # client never sees; say it now, while the client is listening
    if getattr(generator, "use_cuda", False):
        import torch

        if not torch.cuda.is_available():
            messages.append(
                "use_cuda=True but this server has no CUDA device; running on CPU"
            )
    return generator, messages


@router.post("/sessions", response_model=SessionResponse)
def create_session(request: CreateSessionRequest) -> SessionResponse:
    generator, caught = _build_generator(request.generator_name, request.state)
    session_id = _register_session(generator, request.generator_name)
    return SessionResponse(session_id=session_id, warnings=caught)


@router.get("/sessions", response_model=SessionListResponse)
def list_sessions() -> SessionListResponse:
    """All live sessions, so an operator can find and delete orphans."""
    with _lock:
        items = list(_sessions.items())
    infos = []
    for session_id, session in items:
        data = session.generator.data
        infos.append(
            SessionInfo(
                session_id=session_id,
                generator_name=session.generator_name,
                created_at=session.created_at,
                last_used_at=session.last_used_at,
                n_data=0 if data is None else len(data),
            )
        )
    return SessionListResponse(sessions=infos, max_sessions=_max_sessions())


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
        raise _generator_error(exc, 500, "generator failed to produce points")
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
        raise _generator_error(exc, 400, "failed to ingest results")


@router.put("/sessions/{session_id}/data", status_code=204)
def replace_data(session_id: str, request: ReplaceDataRequest) -> None:
    try:
        with _locked_session(session_id) as generator:
            generator.data = pd.DataFrame(request.data)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("replacing generator data failed")
        raise _generator_error(exc, 400, "failed to replace data")


@router.put("/sessions/{session_id}/state", response_model=SessionResponse)
def replace_state(session_id: str, request: ReplaceStateRequest) -> SessionResponse:
    """Rebuild the session's generator from a new configuration, keeping its
    data. The fitted model is dropped and retrained on the next suggest."""
    session = _get_session(session_id)
    with session.lock:
        if session.closed:
            raise HTTPException(
                status_code=404, detail=f"unknown session {session_id!r}"
            )
        generator, caught = _build_generator(session.generator_name, request.state)
        generator.data = session.generator.data
        _finalize_quietly(session.generator)
        session.generator = generator
        session.last_used_at = time.time()
    return SessionResponse(session_id=session_id, warnings=caught)


def state_json(generator: Generator, include_model: bool = True) -> str:
    """The generator's state as JSON with torch modules inline. With
    `include_model=False` the top-level `model` is left out; other torch
    modules (custom kernels, prior means) stay, since they are configuration
    rather than a result of training."""
    text = generator.to_json(serialize_torch=True, serialize_inline=True)
    if include_model:
        return text
    state = json.loads(text)
    state.pop("model", None)
    return json.dumps(state)


@router.get("/sessions/{session_id}/state")
def get_state(session_id: str, include_model: bool = True) -> Response:
    """The generator's full state. `include_model=false` leaves out the fitted
    model, which can be tens of megabytes, for callers that only need the
    configuration and the optimizer state (e.g. a dump that drops the model)."""
    with _locked_session(session_id) as generator:
        text = state_json(generator, include_model)
    return Response(content=text, media_type="application/json")


def _finalize_quietly(generator: Generator) -> None:
    finalize = getattr(generator, "finalize", None)
    if callable(finalize):
        try:
            finalize()
        except Exception:
            logger.exception("generator.finalize failed")


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str) -> None:
    with _lock:
        session = _sessions.pop(session_id, None)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown session {session_id!r}")
    with session.lock:
        session.closed = True
        _finalize_quietly(session.generator)


# token-free so deployment probes can call it; auth is on `router`, not the app
health_router = APIRouter()


@health_router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness and compatibility information."""
    import torch

    with _lock:
        n = len(_sessions)
    return HealthResponse(
        status="ok",
        xopt_version=xopt.__version__,
        cuda_available=torch.cuda.is_available(),
        sessions=n,
        max_sessions=_max_sessions(),
    )


app.include_router(router)
app.include_router(health_router)


def main() -> None:
    import uvicorn

    host = os.environ.get("XOPT_HTTP_GENERATOR_HOST", "127.0.0.1")
    port = int(os.environ.get("XOPT_HTTP_GENERATOR_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    main()
