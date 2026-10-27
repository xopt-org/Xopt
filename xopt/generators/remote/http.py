"""Client-side proxy generator that delegates computation to a remote HTTP server."""

import json
import logging
import warnings
from typing import Any, ClassVar, Optional

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

    @field_serializer("generator")
    def _serialize_generator(
        self, value: Generator, info: FieldSerializationInfo
    ) -> dict[str, Any]:
        # `name` is a ClassVar and not dumped; _resolve_generator needs it
        # to rebuild the wrapped generator when a saved config is reloaded.
        # Dump `value` itself so subclass fields (e.g. UCB `beta`) are kept.
        return {"name": value.name} | value.model_dump(mode=info.mode)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.api_key:
            self._http.headers["Authorization"] = f"Bearer {self.api_key}"

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        # Xopt.reset_data()/remove_data() assign `generator.data` directly, bypassing
        # add_data(); mirror that full replacement remotely (unless we triggered it).
        if (
            name == "data"
            and not getattr(self, "_syncing", False)
            and getattr(self, "_session_id", None) is not None
        ):
            self._push_full_data(value)

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    def _request(
        self, method: str, path: str, *, timeout: Optional[float] = None, **kwargs: Any
    ) -> requests.Response:
        try:
            response = self._http.request(
                method, self._url(path), timeout=timeout or self.timeout, **kwargs
            )
            response.raise_for_status()
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
            self._session_id = response.json()["session_id"]
        return self._session_id

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
            # route through pandas' JSON encoder first so numpy scalar dtypes
            # survive the plain json.dumps() requests uses internally
            json={"results": json.loads(new_data.to_json(orient="records"))},
        )
        self._syncing = True
        try:
            super().add_data(new_data)
        finally:
            self._syncing = False

    def _push_full_data(self, data: Optional[pd.DataFrame]) -> None:
        records = [] if data is None else json.loads(data.to_json(orient="records"))
        self._request(
            "PUT",
            f"/sessions/{self._session_id}/data",
            json={"data": records},
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
                "could not reach the remote generator server before dumping; "
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
        mirror = generator_class.model_validate_json(response.text)
        # `data` is excluded from the remote state dump; reuse our own synced cache
        mirror.data = self.data
        self.generator = mirror
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
