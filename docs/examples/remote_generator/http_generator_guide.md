# Remote Generators — User Guide

Xopt remote generators allow a scientist to run a Jupyter/Xopt client on one computer while generator operations execute in a separate HTTP backend. The backend can be on the same computer or a different host.

| Interface | Use it for |
| --- | --- |
| `HTTPGenerator` | Standard Xopt remote generator and upstream serialization |
| `CloudpickleHTTPGenerator` | Notebook-defined Python classes and custom model components requiring cloudpickle |

The cloudpickle interface is **only for trusted environments**: deserializing a cloudpickle payload can execute arbitrary Python code.

## 1. Install

In a compatible Python environment on the backend and client, install Xopt's remote dependencies from the Xopt repository root:

```bash
python -m pip install -e ".[remote]"
```

For BAX or custom PyTorch models, install the dependencies needed by the chosen generator. Keep Python and package versions compatible between hosts.

## 2. Start the backend

For a local, trusted development setup:

```bash
python -m uvicorn xopt.generators.remote.cloudpickle_server:app \
  --host 127.0.0.1 --port 8002 --log-level info
```

This application serves upstream HTTP generator operations and the optional cloudpickle extension. Use a secured, reachable service address and configure access controls for cross-host deployments. **Do not expose this experimental cloudpickle server to untrusted networks.** Authentication hardening is still under review.

## 3. Connect the client

Use the standard upstream generator for ordinary remote operations:

```python
from xopt.generators.remote.http import HTTPGenerator
```

For notebook-local classes that require cloudpickle:

```python
from xopt.generators.remote.cloudpickle import CloudpickleHTTPGenerator
```

Configure the remote generator with its underlying generator and the backend URL as shown in the notebooks below. For a local server the URL is `http://127.0.0.1:8002`; for a separate backend, use its reachable address. `127.0.0.1` on the client never points to a different host.

## 4. Run a working example

Start the backend, launch Jupyter on the client, and open a notebook from `docs/examples/remote_generator/`:

| Notebook | Example |
| --- | --- |
| `http_generator_ucb.ipynb` | Remote upper-confidence-bound optimization |
| `http_generator_bax.ipynb` | Remote BAX optimization |
| `http_generator_custom_model.ipynb` | Cloudpickle transport of a notebook-defined custom model |

Start with a fresh notebook kernel. Follow that notebook's configuration for the backend URL and generator construction. Where supported by the notebook, set `XOPT_HTTP_URL` before launching Jupyter.

**Execution boundary:** The remote generator runs generator/session operations on the backend. Notebook-side evaluation, plotting, and computations performed after pulling a model state may still run on the client.

## 5. Pull updated generator state

For `CloudpickleHTTPGenerator`, `pull()` retrieves an updated local mirror of the backend generator. Use this when inspecting or plotting model state on the client. The result is a **snapshot**, not a live shared Python object; pull again to see later server changes.

## Security and reproducibility

- Only accept cloudpickle payloads from trusted clients; deserialization can execute arbitrary code.
- For cross-host deployment, require authentication, restrict network access, and use TLS as appropriate. The current cloudpickle authentication behavior is being strengthened in PR review.
- Pin compatible Python dependencies and full source Git revisions on client and backend.
- The lepton-to-Fedora remote workflow has been validated, but that does not mean the experimental server is hardened for general public access.

For focused regression tests from the Xopt repository root:

```bash
python -m pytest xopt/tests/generators/remote/test_http_generator.py -v --no-cov
```

The test fixture starts its own server; no separate port-8002 server is required for these tests.
