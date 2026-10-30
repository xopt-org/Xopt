"""Client-side proxy generator that delegates computation to a remote HTTP server."""

import json
import logging
import warnings
from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd
import requests
from pydantic import (
    Field,
    FieldSerializationInfo,
    PrivateAttr,
    SerializeAsAny,
    field_serializer,
    model_validator,
)

from xopt.errors import XoptError
from xopt.generator import Generator
from xopt.generators import get_generator_dynamic

logger = logging.getLogger(__name__)


def _records_json(key: str, data: Optional[pd.DataFrame]) -> str:
    """JSON body `{key: rows}` with the rows of `data` as records. Floats are
    emitted with `repr`, so values round-trip exactly; `DataFrame.to_json`
    rounds to 10 decimal places, which turns an emittance of 1e-9 into 0 on the
    server. NaN and inf are kept as JSON tokens, which the server's parser
    accepts (`requests`' own `json=` rejects them)."""
    rows = [] if data is None else data.to_dict(orient="records")
    return json.dumps({key: rows}, default=_scalar)


def _scalar(value: Any) -> Any:
    # numpy values and timestamps from DataFrame.to_dict that json can't handle
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is pd.NA or value is pd.NaT:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


_JSON_HEADERS = {"Content-Type": "application/json"}


def _error_detail(response: requests.Response) -> str:
    """The server's error message: FastAPI's `detail`, else the response body."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if detail is None:
        detail = response.text.strip() or response.reason
    return str(detail)[:500]


_SUPPORTS_FLAGS = (
    "supports_batch_generation",
    "supports_no_objective",
    "supports_single_objective",
    "supports_multi_objective",
    "supports_constraints",
    "supports_discrete_variables",
    "supports_contextual_variables",
)


class HTTPGenerator(Generator):
    """
    Generator that proxies `suggest`/`ingest` to a wrapped generator running
    in a session on a remote `xopt.generators.remote.server` instance.

    `generator` holds the (already-instantiated) generator to run remotely. Its
    full state is sent to the server when the session is created, and never by
    sending a pickled object over the wire - the server reconstructs an instance
    of the same class (looked up by `generator.name` in the local registry) from
    that state.

    Methods/attributes not defined on this class (e.g. `visualize_model` on a
    wrapped BO generator) fall back to the `generator` field, which is replaced
    with a fresh, read-only reconstruction of the remote generator's full state
    (pulled via `pull()`) on every such access. Because `generator` is a regular
    field, serializing `HTTPGenerator` itself also captures that latest mirrored
    state. Calling a state-mutating method on the mirror has no effect on the
    remote session.
    """

    name: ClassVar[str] = "http"

    base_url: str = Field(description="base URL of the remote generator server")
    generator: SerializeAsAny[Generator] = Field(
        description="generator to run remotely; also mirrors the latest "
        "pulled remote state after suggest/ingest/pull calls"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="bearer token sent as the Authorization header; "
        "excluded from serialized output so it never lands in dump files",
        exclude=True,
        repr=False,
    )
    timeout: float = Field(
        default=30.0, description="timeout in seconds for suggest/ingest requests"
    )
    state_timeout: float = Field(
        default=120.0,
        description="timeout in seconds for full-state pulls (may include a "
        "serialized fitted model)",
    )
    pull_on_dump: bool = Field(
        default=True,
        description="pull the latest remote state before Xopt.dump() writes a "
        "dump file, so the dump matches the server",
    )
    dump_pull_timeout: float = Field(
        default=10.0,
        description="timeout in seconds for the pull before a dump; if it fails, "
        "the dump is written from the last pulled copy with a warning",
    )

    _session_id: Optional[str] = PrivateAttr(default=None)
    _http: requests.Session = PrivateAttr(default_factory=requests.Session)
    _syncing: bool = PrivateAttr(default=False)

    @model_validator(mode="before")
    @classmethod
    def _resolve_generator(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "generator" not in data:
            return data

        wrapped = data["generator"]
        if isinstance(wrapped, dict):
            wrapped = dict(wrapped)
            try:
                name = wrapped.pop("name")
            except KeyError:
                raise ValueError(
                    "generator dict config must include a 'name' key"
                ) from None
            # Xopt only moves a top-level `vocs` into this outer generator;
            # pass it down so YAML configs don't need to repeat it
            if "vocs" not in wrapped and "vocs" in data:
                wrapped["vocs"] = data["vocs"]
            wrapped = get_generator_dynamic(name).model_validate(wrapped)
            data["generator"] = wrapped

        if isinstance(wrapped, Generator):
            # lets validate_vocs run correctly for the wrapped generator without
            # requiring the caller to duplicate vocs/supports_* flags by hand
            data.setdefault("vocs", wrapped.vocs)
            for flag in _SUPPORTS_FLAGS:
                data.setdefault(flag, getattr(wrapped, flag))

        return data

    @model_validator(mode="after")
    def _check_vocs_match(self) -> "HTTPGenerator":
        # Xopt validates inputs against the outer vocs while the server optimizes
        # with the inner one; a dump holds both, so an edit to one of them would
        # otherwise go unnoticed
        if self.vocs != self.generator.vocs:
            raise ValueError(
                "`vocs` of the HTTPGenerator and of the wrapped generator differ"
            )
        return self

    @field_serializer("generator")
    def _serialize_generator(
        self, value: Generator, info: FieldSerializationInfo
    ) -> dict[str, Any]:
        # `name` is a ClassVar and not dumped; _resolve_generator needs it
        # to rebuild the wrapped generator when a saved config is reloaded.
        # Dump `value` itself so subclass fields (e.g. UCB `beta`) are kept.
        return {"name": value.name} | value.model_dump(mode=info.mode)

    def __deepcopy__(self, memo: dict) -> "HTTPGenerator":
        # Xopt deepcopies its generator on construction; the copy must not share
        # the original's session, or both would ingest into one remote generator
        # and finalize() on either would close it for both
        private = self.__pydantic_private__
        http, private["_http"] = private["_http"], None
        try:
            copied = super().__deepcopy__(memo)
        finally:
            private["_http"] = http
        copied._session_id = None
        copied._http = requests.Session()
        return copied

    def __setattr__(self, name: str, value: Any) -> None:
        # Xopt.reset_data()/remove_data() assign `generator.data` directly, bypassing
        # add_data(); mirror that full replacement remotely (unless we triggered it).
        # Push first so a failed request leaves the local data unchanged.
        if (
            name == "data"
            and not getattr(self, "_syncing", False)
            and getattr(self, "_session_id", None) is not None
        ):
            if value is not None and not isinstance(value, pd.DataFrame):
                value = pd.DataFrame(value)
            self._push_full_data(value)
        super().__setattr__(name, value)

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    def _request(
        self, method: str, path: str, *, timeout: Optional[float] = None, **kwargs: Any
    ) -> requests.Response:
        # set per request so `api_key` assigned after construction takes effect
        headers = dict(kwargs.pop("headers", None) or {})
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = self._http.request(
                method,
                self._url(path),
                timeout=timeout or self.timeout,
                headers=headers,
                **kwargs,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            # include the server's reason (e.g. "unknown generator 'foo'"); the
            # HTTPError message itself is only the status line and URL
            raise XoptError(
                f"HTTPGenerator request to {path!r} failed "
                f"({exc.response.status_code}): {_error_detail(exc.response)}"
            ) from exc
        except requests.RequestException as exc:
            raise XoptError(f"HTTPGenerator request to {path!r} failed: {exc}") from exc
        return response

    def _ensure_session(self) -> str:
        if self._session_id is None:
            state = json.loads(
                self.generator.to_json(serialize_torch=True, serialize_inline=True)
            )
            response = self._request(
                "POST",
                "/sessions",
                json={"generator_name": self.generator.name, "state": state},
            )
            self._start_session(response.json()["session_id"])
        return self._session_id

    def _start_session(self, session_id: str) -> None:
        """Adopt a newly created session and give it the client's data. `data`
        is not part of the generator state, so a new session (first use, after
        finalize(), or on a deepcopy) would otherwise start empty. If the push
        fails the session is dropped again, so the next call retries from
        scratch instead of continuing against a server missing the data."""
        self._session_id = session_id
        if self.data is None or len(self.data) == 0:
            return
        try:
            self._push_full_data(self.data)
        except XoptError:
            self._session_id = None
            try:
                self._request("DELETE", f"/sessions/{session_id}")
            except XoptError:
                pass
            raise

    def generate(self, n_candidates: int) -> list[dict]:
        session_id = self._ensure_session()
        response = self._request(
            "POST",
            f"/sessions/{session_id}/suggest",
            json={"num_points": n_candidates},
        )
        return response.json()["points"]

    def add_data(self, new_data: pd.DataFrame) -> None:
        session_id = self._ensure_session()
        self._request(
            "POST",
            f"/sessions/{session_id}/ingest",
            data=_records_json("results", new_data),
            headers=_JSON_HEADERS,
        )
        self._syncing = True
        try:
            super().add_data(new_data)
        finally:
            self._syncing = False

    def _push_full_data(self, data: Optional[pd.DataFrame]) -> None:
        self._request(
            "PUT",
            f"/sessions/{self._session_id}/data",
            data=_records_json("data", data),
            headers=_JSON_HEADERS,
        )

    def finalize(self) -> None:
        if self._session_id is None:
            return
        try:
            self._request("DELETE", f"/sessions/{self._session_id}")
        except XoptError as exc:
            warnings.warn(f"failed to finalize remote generator session: {exc}")
        finally:
            self._session_id = None

    def sync_for_dump(self) -> None:
        # with no session there is no remote state newer than the local copy
        if not self.pull_on_dump or self._session_id is None:
            return
        try:
            self.pull(timeout=self.dump_pull_timeout)
        except XoptError as exc:
            warnings.warn(
                "could not pull the remote generator state before dumping; "
                "the dump uses the last pulled copy of the generator, which may "
                f"be out of date: {exc}"
            )

    def pull(self, timeout: Optional[float] = None) -> Generator:
        """Fetch the remote generator's full state, rebuild it locally, and store
        it in `self.generator`. `timeout` defaults to `state_timeout`."""
        session_id = self._ensure_session()
        response = self._request(
            "GET",
            f"/sessions/{session_id}/state",
            timeout=timeout or self.state_timeout,
        )
        generator_class = get_generator_dynamic(self.generator.name)
        try:
            mirror = generator_class.model_validate_json(response.text)
            # `data` is excluded from the remote state dump; reuse our own synced cache
            mirror.data = self.data
            self.generator = mirror
        except Exception as exc:
            # e.g. a torch.load failure, or a validator on assignment; raise
            # XoptError so sync_for_dump() can fall back to the last pulled copy
            # instead of breaking the dump
            raise XoptError(
                f"failed to rebuild the remote generator state locally: {exc}"
            ) from exc
        return mirror

    def __getattr__(self, name: str) -> Any:
        try:
            # pydantic resolves private attrs (e.g. _session_id) through its own
            # __getattr__; without this delegation we'd shadow that lookup entirely
            return super().__getattr__(name)
        except AttributeError:
            pass
        if name.startswith("_"):
            raise AttributeError(name)
        # always pull fresh rather than reusing the previous `self.generator`,
        # otherwise a mirror cached from an earlier access would silently go
        # stale as more data is ingested
        return getattr(self.pull(), name)
