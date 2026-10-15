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

Only the first `add_data` call answering a pending request sends a response to
scipy. Repeated submissions of the same candidate remain in the data history,
but cannot answer a later request. Values are also cached under the requested
coordinates, so small readback differences accepted by sequential point
validation do not cause cache misses after a session restart.

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

- `reset()` stops the active worker session and clears the result and completion state.
- `set_data(...)` stops any active session, clears completion state, reloads data, and rebuilds the cache.
- Each worker retains its own stop event, queues, and cache. Shutdown waits up to two seconds and warns if the worker is still running in scipy; that worker cannot consume responses or publish requests, results, or errors into a subsequent session.
- `result` exposes scipy's `OptimizeResult` after termination, including `success`, `message`, and solver-specific fields such as `nit`. It is `None` before termination.
- `is_done` is true once scipy terminates, including unsuccessful termination (for example, reaching an iteration limit). Check `result.success` to distinguish successful convergence.
- Once terminated, later `generate()` calls do not restart scipy or rebuild the cache. For compatibility, they still return the latest known data row, with a `RuntimeWarning` emitted once per session on this fallback path. Further explicit `Xopt.step()` calls will re-evaluate and append this duplicate; stop stepping when `generator.is_done` is true.
- Common scipy `ValueError` messages are remapped to clearer runtime errors (for example unsupported solver availability or bound handling).

## State Restoration

For model-level roundtrips, use pydantic serialization:

- `model_dump()`
- `model_validate(...)`

Then restore the evaluation history with `set_data(...)` so the cache and last outcome are reconstructed before continuing optimization.

The class also defines `__getstate__` and `__setstate__` for explicit state handling of non-picklable runtime thread objects.
Pickling does not stop the original worker. Pickle and deep-copy roundtrips retain
completed results and the completion latch without sharing runtime thread objects;
model-level roundtrips restart from the restored evaluation history instead.
Requested-coordinate cache entries are retained through `reset()`, pickle, and
deep-copy roundtrips. Replacing the dataset with `set_data(...)` clears these
entries; model-level roundtrips rebuild from observed coordinates in the restored
history.

## Configuration

Typical fields:

- `method`: scipy minimization method name, e.g. `Powell`, `Nelder-Mead`, `L-BFGS-B`.
- `initial_point`: optional starting point dictionary.
- `tol`, `options`: passed directly to `scipy.optimize.minimize`.
- `scipy_kwargs`: additional keyword arguments forwarded to scipy.

::: xopt.generators.sequential.scipy
