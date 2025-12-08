# Xopt Stopping Conditions

This module provides a comprehensive set of stopping conditions for Xopt optimization processes. Each stopping condition takes an Xopt dataframe and VOCS object to determine when optimization should stop.

## Available Stopping Conditions

### Basic Conditions

#### `MaxEvaluationsCondition`
Stops after a maximum number of evaluations.

```python
from xopt import MaxEvaluationsCondition

condition = MaxEvaluationsCondition(max_evaluations=100)
```

#### `TargetValueCondition`
Stops when an objective reaches a target value within tolerance.

```python
condition = TargetValueCondition(
    objective_name="f1",
    target_value=0.001,
    tolerance=1e-6
)
```

### Convergence Conditions

#### `ConvergenceCondition`
Stops when improvement falls below a threshold for a specified patience.

```python
condition = ConvergenceCondition(
    objective_name="f1",
    improvement_threshold=1e-6,
    patience=10,
    relative=False  # Use absolute improvement
)
```

#### `StagnationCondition`
Stops when the best objective value hasn't improved for a number of evaluations.

```python
condition = StagnationCondition(
    objective_name="f1",
    patience=20,
    tolerance=1e-8
)
```

#### `RelativeImprovementCondition`
Stops when relative improvement falls below a threshold.

```python
condition = RelativeImprovementCondition(
    objective_name="f1",
    relative_threshold=0.01,  # 1% improvement
    window_size=10
)
```

### Statistical Conditions

#### `VarianceCondition`
Stops when the variance of recent objective values falls below threshold.

```python
condition = VarianceCondition(
    objective_name="f1",
    variance_threshold=1e-6,
    window_size=10
)
```

#### `ObjectiveThresholdCondition`
Stops when an objective crosses a threshold value.

```python
condition = ObjectiveThresholdCondition(
    objective_name="f1",
    threshold=0.5,
    direction="below"  # "above", "below", or "either"
)
```

### Constraint-Based Conditions

#### `FeasibilityCondition`
Stops when a feasible solution is found.

```python
condition = FeasibilityCondition(
    require_all_constraints=True
)
```

### Composite Conditions

#### `CompositeCondition`
Combines multiple stopping conditions with AND/OR logic.

```python
condition = CompositeCondition(
    conditions=[
        MaxEvaluationsCondition(max_evaluations=100),
        TargetValueCondition(objective_name="f1", target_value=0.001),
        ConvergenceCondition(objective_name="f1", improvement_threshold=1e-6, patience=10)
    ],
    logic="or"  # Stop if ANY condition is met
)
```

## Usage Example

```python
from xopt import Xopt, VOCS, Evaluator
from xopt.stopping_conditions import TargetValueCondition, CompositeCondition
from xopt.generators.sequential.neldermead import NelderMeadGenerator

# Define optimization problem
def test_function(input_dict):
    x1, x2 = input_dict["x1"], input_dict["x2"]
    return {"f1": (x1 - 0.5)**2 + (x2 - 0.3)**2}

vocs = VOCS(
    variables={"x1": [0.0, 1.0], "x2": [0.0, 1.0]},
    objectives={"f1": "MINIMIZE"}
)

# Set up Xopt
evaluator = Evaluator(function=test_function)
generator = NelderMeadGenerator(vocs=vocs)
X = Xopt(vocs=vocs, evaluator=evaluator, generator=generator, max_evaluations=100)

# Define stopping condition
stopping_condition = TargetValueCondition(
    objective_name="f1",
    target_value=0.001,
    tolerance=1e-6
)

# Run optimization with stopping condition
X.random_evaluate(1)
while not stopping_condition.should_stop(X.data, X.vocs):
    X.step()

print(f"Optimization stopped after {len(X.data)} evaluations")
```

## Custom Stopping Conditions

You can create custom stopping conditions by inheriting from `StoppingCondition`:

```python
from xopt.stopping_conditions import StoppingCondition

class CustomCondition(StoppingCondition):
    threshold: float
    
    def should_stop(self, data: pd.DataFrame, vocs: VOCS) -> bool:
        # Implement your custom logic here
        return len(data) > 50 and data["f1"].min() < self.threshold
```

## Integration with Xopt

While Xopt has a built-in `max_evaluations` parameter, stopping conditions provide much more flexibility and can be combined in sophisticated ways. The stopping conditions operate on:

- **data**: The Xopt dataframe containing all evaluation results with columns for variables, objectives, constraints, and observables
- **vocs**: The VOCS object defining the optimization problem structure

All stopping conditions use the VOCS object to understand the problem structure and correctly interpret the data columns.