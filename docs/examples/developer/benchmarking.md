# Benchmarking & Profiling

Xopt includes some basic benchmarking tools. This is useful for
comparing configurations and settings.

## Quick Start

Run benchmarks from the command line using `bench_runner.py`:

```bash
python -m xopt.resources.bench_runner bench_build_standard bench_build_batched -n 5 -device cpu
```

This runs the named benchmarks 5 times each on CPU and prints a results table.

To profile with `py-spy` (generates a flamegraph):

```bash
python -m xopt.resources.bench_profiler bench_build_standard -n 3 -device cpu
```

## Framework Components

### BenchSuite

`BenchSuite` manages a collection of benchmark configurations and runs them
with timing statistics. It supports minimum time/round constraints, warmup
runs, and optional GC disabling.

```python
from xopt.resources.bench_framework import BenchSuite

suite = BenchSuite()
suite.add(my_function, kwargs={"n": 100})
suite.run(min_rounds=5, min_time=1.0)
```

Output includes per-function timing: average, median, min, max, total, and
standard deviation.

### BenchDispatcher

A registry for benchmark functions using decorators. Registered benchmarks
can be discovered and run by the CLI runner.

```python
from xopt.resources.bench_framework import BenchDispatcher

@BenchDispatcher.register_decorator()
def bench_my_operation(device="cpu"):
    # setup and run
    ...
```

Default arguments can be registered separately:

```python
@BenchDispatcher.register_defaults(["vocs", "data"], lambda: make_test_data())
@BenchDispatcher.register_decorator()
def bench_my_operation(vocs, data, device="cpu"):
    ...
```

### Available Benchmarks

Benchmark functions are defined in `xopt/resources/bench_functions/`. Current
modules:

- **`models.py`** - GP model construction benchmarks (standard vs batched,
  LBFGS vs Adam vs GPyTorch optimizers)
- **`generators.py`** - End-to-end generator step benchmarks

List all registered benchmarks by importing the module:

```python
import xopt.resources.bench_functions  # registers all benchmarks
from xopt.resources.bench_framework import BenchDispatcher

print(list(BenchDispatcher.benchmarks.keys()))
```

## Utility Functions

- `time_call(f, n)` - Time a callable over `n` repetitions
- `profile_function(func)` - Decorator that profiles a function with `cProfile`
- `generate_vocs(n_vars, n_obj, n_constr)` - Generate synthetic VOCS for benchmarks
- `generate_data(vocs, n)` - Generate synthetic data matching a VOCS definition
