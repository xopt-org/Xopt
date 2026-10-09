# Xopt HTTPGenerator: independent-process and cloudpickle experiments

## Purpose

The goal is a **distributed client-server workflow**: scientists run Jupyter/Xopt on a client-facing machine, while an HTTPGenerator backend runs independently on another compute node (eventually in Docker or Kubernetes). A two-process test on one host is the first validation step; it is **not** proof of cross-node deployment.

The experiments also investigate transporting a notebook-defined PyTorch `ConstraintPrior` without installing a shared scientist-written module on the server. This uses **opt-in cloudpickle**, not a replacement for standard upstream HTTPGenerator serialization.

## Validated notebook results (local workstation, independent Uvicorn process)

| Notebook | Transport | Observed result |
| --- | --- | --- |
| `http_generator_ucb_independent.ipynb` | Upstream HTTPGenerator | Remote UCB session finalized |
| `http_generator_bax_tutorial.ipynb` | Upstream HTTPGenerator | Remote BAX session finalized |
| `http_generator_custom_model_cloudpickle_upstream.ipynb` | Experimental cloudpickle extension | Remote suggested point; recovered `ConstraintPrior`; session finalized |

All three ran against the same server on **port 8002**. These are manually observed notebook results, not automated pytest results. The numerical suggestion may vary between runs.

## Prerequisites

Use the Xopt `http-generator` branch and the `xopt-http-dev` environment with editable Xopt, its remote dependencies, PyTorch/BoTorch/GPyTorch, `cloudpickle`, JupyterLab, and plotting packages. The experiment was initially based on commit `460e28da`.

From the Xopt repository root, install the experimental package:

```bash
conda activate xopt-http-dev
python -m pip install -e ./docs/examples/remote_cloudpickle
```

An editable installation makes `cloudpickle_xopt` importable from other working directories. If you move the source directory, reinstall it.

## Local validation: two independent processes

**Terminal 1 — server:**

```bash
conda activate xopt-http-dev
python -m uvicorn cloudpickle_xopt.server:app \
    --host 127.0.0.1 --port 8002 --log-level info
```

**Terminal 2 — client:**

```bash
conda activate xopt-http-dev
curl -fsS http://127.0.0.1:8002/openapi.json >/dev/null && echo 'Server ready'
python -m jupyter lab
```

Open the three notebooks in the table above and run each from a restarted kernel. They use `http://127.0.0.1:8002` by default. For a different endpoint, set `XOPT_HTTP_URL` before launching Jupyter, where supported by the notebook:

```bash
export XOPT_HTTP_URL=http://127.0.0.1:8002
```

UCB and BAX use `from xopt.generators.remote.http import HTTPGenerator` (standard upstream serialization). Only the custom-prior notebook uses `cloudpickle_xopt.http`. The server extends upstream's FastAPI app and exposes both route families.

**Execution boundary:** HTTPGenerator sends generator/session operations to the server. Notebook-side evaluations, local model training after a state pull, and plots may still run on the client; HTTPGenerator does not automatically move every calculation to the backend.

## Target architecture: two different nodes

The scientist's Jupyter process runs on a workstation or login host; the Uvicorn backend runs in a container on another node. Configure the notebook URL to point to a **reachable, secured backend endpoint** rather than `127.0.0.1`, which always refers to the client's own network namespace. Docker/Kubernetes deployment requires a published service endpoint, compatible server dependencies, and an authentication/network isolation design. Cross-node execution has **not yet been validated**.

Do not expose the experimental cloudpickle endpoint to untrusted clients: `cloudpickle.loads()` can execute arbitrary code. This proof of concept should remain bound to localhost until a deliberate security design exists.

## Automated pytest: Stage 1 and Stage 2

All 14 tests are combined in the original `xopt/tests/generators/remote/test_http_generator.py` file (not a top-level `tests/` directory).

**Stage 1 — independent-process upstream regression:** The original 12 HTTPGenerator tests now launch Uvicorn in a **subprocess**, not a thread. On a local workstation, the patched Stage 1 file completed with **12 passed, 2 warnings in 7.75 s**. This includes a *module-defined* `ConstraintPrior`, which works through upstream serialization. This does not establish that a notebook-local class works.

**Stage 2 — notebook-local prior:** the same test file contains two additional focused tests against the experimental server in its own subprocess:

1. A **negative control** using the upstream HTTPGenerator and a function-local PyTorch `ConstraintPrior`, which should fail to transport successfully. If this unexpectedly succeeds, investigate rather than treating the test as passing.
2. A **positive control** using `cloudpickle_xopt.http.HTTPGenerator` with the same function-local class. Verify a remote suggestion, retrieve the server-side generator, check the recovered class, and numerically compare its cosine prior mean against the expected function.

Install the adapter once, then run from the repository root:

```bash
conda activate xopt-http-dev
python -m pip install -e ./docs/examples/remote_cloudpickle

python -m pytest xopt/tests/generators/remote/test_http_generator.py -v -s --no-cov
```

Pytest automatically starts and stops its own localhost Uvicorn subprocess on an **ephemeral port**. You do **not** need to start port 8002 manually; that fixed port is only for the interactive notebooks. Final validation on a local workstation succeeded: all **14 tests passed** in the consolidated regression suite, and all three notebooks executed successfully against an independent Uvicorn server. These tests validate separate processes on one host, not yet separate hosts or Kubernetes.

Security: cloudpickle deserialization can execute arbitrary code. Use only trusted, isolated localhost processes for this experiment; a distributed deployment needs authentication, isolation, and a deliberate trust boundary.

## Scope and source hygiene

This directory is an experimental adapter and demonstration, not an upstream Xopt source modification. Keep this PR focused on the experimental package, its three validated notebooks, and the existing HTTPGenerator regression test module. Exclude generated `*.egg-info/`, `__pycache__/`, and `.ipynb_checkpoints/` from Git.
