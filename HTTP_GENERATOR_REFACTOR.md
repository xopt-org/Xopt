# HTTPGenerator integration refactor

Source snapshot: `experimental/http-generator-cloudpickle`, commit `70df0038`.

## Changes

- Kept the upstream `xopt.generators.remote.http.HTTPGenerator` and `xopt.generators.remote.server` unchanged.
- Moved the experimental adapter to `xopt.generators.remote.cloudpickle.CloudpickleHTTPGenerator` (with an `HTTPGenerator` alias for early experiments).
- Moved opt-in FastAPI endpoints to `xopt.generators.remote.cloudpickle_server:app`.
- Moved three notebooks to `docs/examples/single_objective_bayes_opt/` with `http_generator_` names.
- Replaced the old thread-based UCB tutorial with the independently hosted UCB tutorial.
- Updated the remote test suite, MkDocs navigation, and `[remote]` optional dependencies.
- Removed the separate `docs/examples/remote_cloudpickle` Python package and its nested `pyproject.toml`.

## Development

```bash
python -m pip install -e '.[remote]'
python -m pytest xopt/tests/generators/remote/test_http_generator.py -v --no-cov
python -m uvicorn xopt.generators.remote.cloudpickle_server:app --host 127.0.0.1 --port 8002
```

Use `XOPT_HTTP_URL=http://127.0.0.1:8002` in the Jupyter environment. The custom-model notebook imports `CloudpickleHTTPGenerator`; the UCB/BAX notebooks use standard `HTTPGenerator`.

## Deployment boundary

Docker Compose and Kubernetes deployment belong in the separate `Xopt-remote` repository, installing this Xopt fork at a pinned Git SHA. Do not vendor Xopt into that repository. The cloudpickle server deserializes executable Python objects: only expose it to trusted, isolated clients; bind localhost for development.

## Validation

Refactor checked for Python syntax, notebook JSON validity, and file references. **No full Xopt pytest run, notebook execution, Docker build, or Kubernetes test has been performed here.** Run these in the `xopt-http-dev` environment before committing.
