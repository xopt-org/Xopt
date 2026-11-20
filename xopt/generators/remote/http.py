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

from xopt.errors import DataError, XoptError
from xopt.generator import Generator, StateOwner
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


def _body(key: str, data: Optional[pd.DataFrame]) -> dict[str, Any]:
    """`requests` kwargs for a JSON body of records."""
    return {"data": _records_json(key, data), "headers": _JSON_HEADERS}


def _config_state(generator: Generator) -> dict[str, Any]:
    """The generator's state with torch modules inline but without the fitted
    `model`: what `push()` sends and what a recovered session is built from.
    Custom kernels and prior means are configuration and stay in."""
    state = json.loads(generator.to_json(serialize_torch=True, serialize_inline=True))
    state.pop("model", None)
    return state


def _fingerprint(generator: Generator) -> str:
    """The settings of `generator` as a string, to detect local edits that were
    not pushed. Torch modules are left out: pickling them fails for classes
    defined in a notebook (which only cloudpickle can send), and an edit to a
    kernel's hyperparameters is not a configuration change."""
    return generator.to_json(serialize_torch=False)


class _TransientError(XoptError):
    """A request whose outcome is unknown or that the server could not handle
    (timeout, connection error, 5xx). Safe to repair with an idempotent PUT."""


def _error_detail(response: requests.Response) -> str:
    """The server's error message: FastAPI's `detail`, else the response body."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if detail is None:
        detail = response.text.strip() or getattr(response, "reason", None) or ""
    return str(detail)[:500]


class RemoteSessionLost(XoptError):
    """The server no longer knows the client's session (restart, eviction, or
    deleted by hand). `HTTPGenerator` recovers from this by creating a new one."""


def _raise_for_response(path: str, response: requests.Response) -> None:
    """Translate an error response into the exception a local generator would
    have raised: `DataError` when the server tagged one, `RemoteSessionLost`
    for an unknown session, `XoptError` otherwise."""
    status = response.status_code
    try:
        body = response.json()
    except ValueError:
        body = None
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict) and detail.get("type") == "DataError":
        raise DataError(detail.get("message", ""))
    message = (
        f"HTTPGenerator request to {path!r} failed ({status}): "
        f"{_error_detail(response)}"
    )
    if status == 404 and "unknown session" in str(detail):
        raise RemoteSessionLost(message)
    if status >= 500:
        raise _TransientError(message)
    raise XoptError(message)


def _reject_unsupported(generator: Generator) -> None:
    """Refuse generators whose state would be silently altered by the trip to
    the server. See the user guide, "Limitations and next steps"."""
    if isinstance(generator, StateOwner):
        raise ValueError(
            f"{type(generator).__name__} cannot run remotely yet: reloading a dump "
            "re-ingests its data and its private state is not serialized"
        )
    if getattr(generator, "custom_objective", None) is not None:
        raise ValueError(
            "a generator with `custom_objective` cannot run remotely yet: the "
            "server has no decoder for that field"
        )
    cost_function = getattr(generator, "cost_function", None)
    if cost_function is not None:
        default = type(generator).model_fields["cost_function"].default
        if cost_function is not default:
            raise ValueError(
                "a custom `cost_function` cannot run remotely yet: the field is "
                "not serialized, so the server would use the default"
            )


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

    The server owns the generator. `generator` holds the configuration sent
    when the session is created and, after `pull()`, a read-only copy of the
    server's state including the fitted model. The server reconstructs the
    class by `generator.name` from its registry; no pickled object crosses
    the wire. Assigning a field on the copy warns, because it changes nothing
    on the server until `push()` sends it.

    The client owns the data. `data` is pushed to the server on every
    `add_data` and whenever it is replaced, and a new session always starts
    from the data the client holds. If the server forgets the session (a
    restart), the next call recreates it from the last pulled copy and the
    local data, with a warning.

    Attributes of the wrapped generator class that this class lacks, such as
    `visualize_model`, resolve against a fresh `pull()`.
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
        default=300.0,
        description="timeout in seconds for suggest/ingest requests; a GP fit on "
        "a shared server can take minutes",
    )
    recover_session: bool = Field(
        default=True,
        description="when the server no longer knows the session, create a new "
        "one from the last pulled state and the local data instead of raising",
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
    # set while `generator` is replaced by this class, so the assignment warning
    # in `_warn_on_mirror_edit` only fires for user edits
    _updating_mirror: bool = PrivateAttr(default=False)
    _steps: int = PrivateAttr(default=0)
    # step count at the last pull that included the model; delegated attribute
    # access reuses the mirror while this equals `_steps`
    _full_pull_step: Optional[int] = PrivateAttr(default=None)
    # the configuration the server is known to have: set by pull() and push(),
    # used to recreate a lost session and to describe what it resumed from
    _synced_state: Optional[dict[str, Any]] = PrivateAttr(default=None)
    _synced_at_step: Optional[int] = PrivateAttr(default=None)
    _fingerprint: Optional[str] = PrivateAttr(default=None)
    _recovering: bool = PrivateAttr(default=False)

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
        _reject_unsupported(self.generator)
        if self._fingerprint is None:
            self._fingerprint = _fingerprint(self.generator)
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
        if name == "generator" and not getattr(self, "_updating_mirror", False):
            # a new local configuration, not yet on the server: it is what a new
            # session is built from, and until pushed the live session differs
            self._full_pull_step = None
            self._synced_state = None
            self._synced_at_step = None
            if self._fingerprint is not None and self._session_id is not None:
                self._fingerprint = ""  # differs from any real fingerprint
            else:
                self._fingerprint = _fingerprint(value)

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
        except requests.RequestException as exc:
            raise _TransientError(
                f"HTTPGenerator request to {path!r} failed: {exc}"
            ) from exc
        # checked here rather than with raise_for_status() so the message carries
        # the server's reason (e.g. "unknown generator 'foo'"), and so a client
        # that only mimics the response (status_code, json(), text, reason)
        # behaves the same
        if response.status_code >= 400:
            _raise_for_response(path, response)
        return response

    def _call(
        self, method: str, path_template: str, **kwargs: Any
    ) -> requests.Response:
        """A request against the session, created on demand. If the server has
        forgotten the session, recreate it and retry once."""
        session_id = self._ensure_session()
        try:
            return self._request(method, path_template.format(session_id), **kwargs)
        except RemoteSessionLost as exc:
            if not self.recover_session or self._recovering:
                raise
            self._session_id = None
            warnings.warn(
                f"the remote generator session is gone ({exc}); recreating it "
                f"from {self._synced_description()} and the local data"
            )
            self._recovering = True
            try:
                session_id = self._ensure_session()
                return self._request(method, path_template.format(session_id), **kwargs)
            finally:
                self._recovering = False

    def _synced_description(self) -> str:
        if self._synced_at_step is None:
            return (
                "the initial configuration (nothing pulled or pushed since; state "
                "the server built up, such as a TuRBO trust region, is lost)"
            )
        return f"the state synced after step {self._synced_at_step}"

    def _warn_if_mirror_edited(self) -> None:
        if (
            self._fingerprint is not None
            and _fingerprint(self.generator) != self._fingerprint
        ):
            warnings.warn(
                "the local copy of the remote generator was changed but not sent to "
                f"the server; {self._apply_edit_hint()}"
            )
            # warn once per edit
            self._fingerprint = _fingerprint(self.generator)

    def _apply_edit_hint(self) -> str:
        return "call `push()` to apply the change, or `pull()` to discard it"

    def _session_state(self) -> dict[str, Any]:
        """The configuration a new session is built from: what the server last
        had, when known, so unpushed local edits do not slip in on recovery."""
        if self._synced_state is not None:
            return self._synced_state
        return json.loads(
            self.generator.to_json(serialize_torch=True, serialize_inline=True)
        )

    def _ensure_session(self) -> str:
        if self._session_id is None:
            state = self._session_state()
            response = self._request(
                "POST",
                "/sessions",
                json={"generator_name": self.generator.name, "state": state},
            )
            body = response.json()
            self._start_session(body["session_id"], body.get("warnings", ()))
        return self._session_id

    def _start_session(self, session_id: str, server_warnings=()) -> None:
        """Adopt a newly created session and give it the client's data. `data`
        is not part of the generator state, so a new session (first use, after
        finalize(), or on a deepcopy) would otherwise start empty. If the push
        fails the session is dropped again, so the next call retries from
        scratch instead of continuing against a server missing the data."""
        for message in server_warnings:
            warnings.warn(f"remote generator server: {message}")
        self._session_id = session_id
        if self.data is None or len(self.data) == 0:
            return
        try:
            self._request(
                "PUT", f"/sessions/{session_id}/data", **_body("data", self.data)
            )
        except XoptError:
            self._session_id = None
            try:
                self._request("DELETE", f"/sessions/{session_id}")
            except XoptError:
                pass
            raise

    def generate(self, n_candidates: int) -> list[dict]:
        self._warn_if_mirror_edited()
        response = self._call(
            "POST", "/sessions/{}/suggest", json={"num_points": n_candidates}
        )
        self._steps += 1
        return response.json()["points"]

    def add_data(self, new_data: pd.DataFrame) -> None:
        self._warn_if_mirror_edited()
        try:
            self._call("POST", "/sessions/{}/ingest", **_body("results", new_data))
        except _TransientError as exc:
            # the ingest may or may not have been applied (e.g. a timeout), so
            # don't retry it: replace the server's data with what the client
            # will hold, which gives the same result either way
            logger.warning("ingest failed (%s); replacing the remote data instead", exc)
            combined = (
                new_data
                if self.data is None
                else pd.concat([self.data, new_data], ignore_index=True)
            )
            self._push_full_data(combined)
        self._syncing = True
        try:
            super().add_data(new_data)
        finally:
            self._syncing = False
        self._full_pull_step = None

    def _push_full_data(self, data: Optional[pd.DataFrame]) -> None:
        self._call("PUT", "/sessions/{}/data", **_body("data", data))

    def push(self) -> None:
        """Send the local configuration (`generator`) to the server's session.
        The server keeps its data, drops the fitted model and retrains on the
        next step. With no session yet, the configuration is sent when the
        session is created, so there is nothing to do."""
        self._fingerprint = _fingerprint(self.generator)
        if self._session_id is None:
            return
        if self._steps and self._synced_at_step != self._steps:
            # the copy may predate state the server built up (a TuRBO trust
            # region, acquisition history); pushing it would roll that back
            warnings.warn(
                "push() sends the local copy's optimizer state as well as its "
                "settings; the copy was not pulled after the last step, so state "
                "the server built up since is replaced by the copy's"
            )
        state = _config_state(self.generator)
        response = self._call("PUT", "/sessions/{}/state", json={"state": state})
        for message in response.json().get("warnings", ()):
            warnings.warn(f"remote generator server: {message}")
        self._synced_state = state
        self._synced_at_step = self._steps
        self._full_pull_step = None

    def finalize(self) -> None:
        if self._session_id is None:
            return
        try:
            self._request("DELETE", f"/sessions/{self._session_id}")
        except XoptError as exc:
            warnings.warn(f"failed to finalize remote generator session: {exc}")
        finally:
            self._session_id = None

    def sync_for_dump(self, include_model: bool = True) -> None:
        # with no session there is no remote state newer than the local copy
        if not self.pull_on_dump or self._session_id is None:
            return
        try:
            self.pull(timeout=self.dump_pull_timeout, include_model=include_model)
        except XoptError as exc:
            warnings.warn(
                "could not pull the remote generator state before dumping; "
                "the dump uses the last pulled copy of the generator, which may "
                f"be out of date: {exc}"
            )

    def pull(
        self, timeout: Optional[float] = None, include_model: bool = True
    ) -> Generator:
        """Fetch the remote generator's state, rebuild it locally and store it
        in `generator`. `include_model=False` leaves out the fitted model,
        which can be large. `timeout` defaults to `state_timeout`."""
        response = self._call(
            "GET",
            "/sessions/{}/state",
            params={"include_model": "true" if include_model else "false"},
            timeout=timeout or self.state_timeout,
        )
        generator_class = get_generator_dynamic(self.generator.name)
        try:
            mirror = generator_class.model_validate_json(response.text)
            # `data` is excluded from the remote state dump; reuse our own synced cache
            mirror.data = self.data
            self._updating_mirror = True
            try:
                self.generator = mirror
            finally:
                self._updating_mirror = False
        except Exception as exc:
            # e.g. a torch.load failure, or a validator on assignment; raise
            # XoptError so sync_for_dump() can fall back to the last pulled copy
            # instead of breaking the dump
            raise XoptError(
                f"failed to rebuild the remote generator state locally: {exc}"
            ) from exc
        self._synced_state = _config_state(mirror)
        self._synced_at_step = self._steps
        self._fingerprint = _fingerprint(mirror)
        # a copy without the model does not stand in for a full pull
        self._full_pull_step = self._steps if include_model else None
        return mirror

    def __getattr__(self, name: str) -> Any:
        try:
            # pydantic resolves private attrs (e.g. _session_id) through its own
            # __getattr__; without this delegation we'd shadow that lookup entirely
            return super().__getattr__(name)
        except AttributeError:
            pass
        # only names the wrapped class defines are delegated, so `hasattr`
        # probes and attributes of this class that failed internally never
        # reach the server
        # `generator` is missing on a model_construct()ed instance
        wrapped = self.__dict__.get("generator")
        if (
            name.startswith("_")
            or wrapped is None
            or hasattr(type(self), name)
            or not (hasattr(type(wrapped), name) or name in type(wrapped).model_fields)
        ):
            raise AttributeError(name)
        if self._session_id is None:
            # nothing on the server is newer than the local configuration
            return getattr(self.generator, name)
        # one pull per step: reuse the mirror until the server's state moved on
        if self._full_pull_step != self._steps:
            self.pull()
        return getattr(self.generator, name)
