# Scipy Minimize Generator

`ScipyGenerator` exposes scipy's `optimize.minimize` methods through Xopt's sequential ask/tell interface.

## Integration Model

Xopt evaluates objective functions externally, one point at a time. `scipy.optimize.minimize` expects an in-process callable objective. `ScipyGenerator` bridges this mismatch by replaying known evaluations:

1. A cache is built from `X.data`.
2. `minimize` is called with an objective wrapper that checks the cache first.
3. If scipy asks for an unseen point, the generator raises an internal signal, exits `minimize`, and returns that point to Xopt.
4. Xopt evaluates that point and appends the result.
5. On the next `step`, `minimize` is called again with the larger cache.

## Performance Notes

- Cache reconstruction is O(N) per `step`, where N is the number of collected evaluations.
- `minimize` restarts each `step`, so there is repeated optimizer bookkeeping overhead.
- For expensive evaluations, this overhead is usually negligible.
- For cheap synthetic test functions, this overhead can be a significant part of runtime.

## Configuration

Typical fields:

- `method`: scipy minimization method name, e.g. `Powell`, `Nelder-Mead`, `L-BFGS-B`.
- `initial_point`: optional starting point dictionary.
- `tol`, `options`: passed directly to `scipy.optimize.minimize`.
- `scipy_kwargs`: additional keyword arguments forwarded to scipy.

::: xopt.generators.sequential.scipy
