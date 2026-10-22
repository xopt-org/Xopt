# Stopping Conditions

## Maximum elapsed time

Use `MaxTimeCondition` to stop after a positive, finite number of wall-clock
seconds:

```python
from xopt.stopping_conditions import MaxTimeCondition

X.stopping_condition = MaxTimeCondition(max_time=60.0)
X.run()
```

The timer starts when `should_stop()` is first called, normally at the start of
`run()`, and uses a monotonic clock. Time before that first check is not counted.
Conditions are checked between optimization steps, so an
in-progress evaluation is allowed to finish and the time limit may be exceeded.
The condition also works without evaluation data and can be nested inside a
`CompositeCondition`.

For YAML configuration:

```yaml
stopping_condition:
  name: MaxTimeCondition
  max_time: 60.0
```

Only the time limit is serialized. Loading a saved configuration resets the
timer; the full time budget starts at the first check.

::: xopt.stopping_conditions
