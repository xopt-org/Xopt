# Scipy Minimize Generator

`ScipyGenerator` exposes scipy's `optimize.minimize` methods through Xopt's sequential ask/tell interface.

## Integration Model

Xopt evaluates objective functions externally, one point at a time. `scipy.optimize.minimize` expects an in-process callable objective. `ScipyGenerator` bridges this mismatch by running one persistent scipy session in a worker thread:

1. A cache is built from prior evaluations (`data`) keyed by rounded variable values.
2. A single `minimize` call starts in a background thread.
3. The objective callback first checks the cache.
4. For an uncached point, that point is pushed to Xopt via a request queue.
5. Xopt evaluates the point externally and calls `add_data`.
6. The objective value is sent back through a response queue, and the same scipy run continues from in-memory state.

## Performance Notes

- Cache reconstruction is O(N) when a session starts or data is reloaded.
- The active scipy run is maintained between `step` calls; `minimize` is not restarted each step.
- Keys are rounded to 12 decimals before cache lookup to reduce floating-point key mismatch issues.

## Supported Methods

`method` is validated against bounded scipy methods supported by this wrapper:

- `Nelder-Mead`
- `Powell`
- `L-BFGS-B`
- `TNC`
- `SLSQP`
- `trust-constr`
- `COBYLA`
- `COBYQA`

Invalid or empty method names fail validation.

## Session Lifecycle and Errors

- `reset()` stops the active worker session and clears transient runtime state.
- `set_data(...)` stops any active session, reloads data, and rebuilds the cache.
- If scipy converges using only cached values, generation falls back to the latest known data row.
- Common scipy `ValueError` messages are remapped to clearer runtime errors (for example unsupported solver availability or bound handling).

## State Restoration

For model-level roundtrips, use pydantic serialization:

- `model_dump()`
- `model_validate(...)`

Then restore the evaluation history with `set_data(...)` so the cache and last outcome are reconstructed before continuing optimization.

The class also defines `__getstate__` and `__setstate__` for explicit state handling of non-picklable runtime thread objects.

## Configuration

Typical fields:

- `method`: scipy minimization method name, e.g. `Powell`, `Nelder-Mead`, `L-BFGS-B`.
- `initial_point`: optional starting point dictionary.
- `tol`, `options`: passed directly to `scipy.optimize.minimize`.
- `scipy_kwargs`: additional keyword arguments forwarded to scipy.

::: xopt.generators.sequential.scipy
