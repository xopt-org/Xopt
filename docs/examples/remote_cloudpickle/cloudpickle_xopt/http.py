"""Opt-in HTTPGenerator adapter for notebook-defined Python classes.

Only session creation and state retrieval use cloudpickle. Suggestion,
ingestion, and finalization continue through the upstream HTTPGenerator API.
The remote endpoint must be trusted: unpickling is code execution.
"""

import base64

import cloudpickle

from xopt.generator import Generator
from xopt.generators.remote.http import HTTPGenerator as UpstreamHTTPGenerator


class HTTPGenerator(UpstreamHTTPGenerator):
    """Use cloudpickle for generator transfer, preserving upstream behavior.

    Scientists can define a custom class in a notebook without installing its
    source module on the backend. Required libraries must still be installed
    and compatible on both sides.
    """

    def _ensure_session(self) -> str:
        """Lazily create one remote session using the opt-in endpoint."""
        if self._session_id is None:
            # Encode arbitrary Python generator state, including local classes.
            # Base64 is for JSON transport only; it provides no security.
            payload = base64.b64encode(cloudpickle.dumps(self.generator)).decode("ascii")
            response = self._request("POST", "/sessions/cloudpickle", json={"payload": payload})
            self._session_id = response.json()["session_id"]
        return self._session_id

    def pull(self) -> Generator:
        """Refresh the local mirror from the server's latest generator state."""
        session_id = self._ensure_session()
        response = self._request(
            "GET", f"/sessions/{session_id}/state/cloudpickle", timeout=self.state_timeout
        )
        # This endpoint is trusted for the same reason as session creation:
        # cloudpickle.loads() can execute Python code in the client process.
        mirror = cloudpickle.loads(base64.b64decode(response.json()["payload"], validate=True))
        if not isinstance(mirror, Generator):
            raise TypeError("server returned a non-Generator object")
        # Preserve upstream's local-data ownership semantics. The generator
        # model/state is remote; the client's collected data remains local.
        mirror.data = self.data
        self.generator = mirror
        return mirror
