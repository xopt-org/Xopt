"""Opt-in HTTPGenerator adapter for notebook-defined Python classes.

Only session creation and state retrieval use cloudpickle. Suggestion,
ingestion, and finalization continue through the upstream HTTPGenerator API.
The remote endpoint must be trusted: unpickling is code execution.
"""

import base64
from typing import ClassVar, Optional

import cloudpickle

from xopt.errors import XoptError
from xopt.generator import Generator
from xopt.generators.remote.http import HTTPGenerator as UpstreamHTTPGenerator


class CloudpickleHTTPGenerator(UpstreamHTTPGenerator):
    """Use cloudpickle for generator transfer, preserving upstream behavior.

    Scientists can define a custom class in a notebook without installing its
    source module on the backend. Required libraries must still be installed
    and compatible on both sides.
    """

    name: ClassVar[str] = "cloudpickle_http"

    def _ensure_session(self) -> str:
        """Lazily create one remote session using the opt-in endpoint."""
        if self._session_id is None:
            # Encode arbitrary Python generator state, including local classes.
            # Base64 is for JSON transport only; it provides no security.
            payload = base64.b64encode(cloudpickle.dumps(self.generator)).decode(
                "ascii"
            )
            response = self._request(
                "POST", "/sessions/cloudpickle", json={"payload": payload}
            )
            body = response.json()
            self._start_session(body["session_id"], body.get("warnings", ()))
        return self._session_id

    def push(self) -> None:
        raise NotImplementedError(
            "CloudpickleHTTPGenerator cannot push a configuration; finalize() and "
            "create a new generator instead"
        )

    def _apply_edit_hint(self) -> str:
        return (
            "CloudpickleHTTPGenerator cannot push changes; call `pull()` to discard "
            "it, or finalize() and create a new generator"
        )

    def _session_state(self) -> dict:
        # the cloudpickle session is created from the live object, not a state
        # dict; recovery resends the current copy
        return {}

    def pull(
        self, timeout: Optional[float] = None, include_model: bool = True
    ) -> Generator:
        """Refresh the local mirror from the server's latest generator state.
        The cloudpickle snapshot always includes the model."""
        response = self._call(
            "GET",
            "/sessions/{}/state/cloudpickle",
            timeout=timeout or self.state_timeout,
        )
        # This endpoint is trusted for the same reason as session creation:
        # cloudpickle.loads() can execute Python code in the client process.
        try:
            mirror = cloudpickle.loads(
                base64.b64decode(response.json()["payload"], validate=True)
            )
            if not isinstance(mirror, Generator):
                raise TypeError("server returned a non-Generator object")
            # Preserve upstream's local-data ownership semantics. The generator
            # model/state is remote; the client's collected data remains local.
            mirror.data = self.data
            self._updating_mirror = True
            try:
                self.generator = mirror
            finally:
                self._updating_mirror = False
        except Exception as exc:
            # XoptError lets sync_for_dump() fall back to the last pulled copy
            raise XoptError(
                f"failed to rebuild the remote generator state locally: {exc}"
            ) from exc
        self._synced_state = None  # recreated from the pickled copy instead
        self._synced_at_step = self._steps
        self._full_pull_step = self._steps
        return mirror
