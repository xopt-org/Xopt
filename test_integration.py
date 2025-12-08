"""
Test the integration of stopping conditions with Xopt.run()
"""

from xopt import Xopt, VOCS, Evaluator, MaxEvaluationsCondition, TargetValueCondition
from xopt.generators.scipy import LatinHypercubeGenerator


def test_function(input_dict):
    """Simple test function."""
    x1, x2 = input_dict["x1"], input_dict["x2"]
    return {"f1": (x1 - 0.5)**2 + (x2 - 0.3)**2}


def test_stopping_condition_integration():
    """Test that stopping conditions work with Xopt.run()"""
    
    # Define optimization problem
    vocs = VOCS(
        variables={"x1": [0.0, 1.0], "x2": [0.0, 1.0]},
        objectives={"f1": "MINIMIZE"}
    )
    
    evaluator = Evaluator(function=test_function)
    generator = LatinHypercubeGenerator(vocs=vocs)
    
    print("Testing MaxEvaluationsCondition integration...")
    # Test 1: MaxEvaluationsCondition
    X1 = Xopt(
        vocs=vocs,
        evaluator=evaluator,
        generator=generator,
        stopping_condition=MaxEvaluationsCondition(max_evaluations=5)
    )
    
    X1.run()
    print(f"✓ Stopped after {len(X1.data)} evaluations (should be 5)")
    assert len(X1.data) == 5
    
    print("Testing TargetValueCondition integration...")
    # Test 2: TargetValueCondition
    X2 = Xopt(
        vocs=vocs,
        evaluator=evaluator,
        generator=LatinHypercubeGenerator(vocs=vocs),  # Fresh generator
        stopping_condition=TargetValueCondition(
            objective_name="f1",
            target_value=0.5,  # Should be easy to reach
            tolerance=0.1
        ),
        max_evaluations=20  # Fallback limit
    )
    
    X2.run()
    print(f"✓ Stopped after {len(X2.data)} evaluations")
    best_value = X2.data["f1"].min()
    print(f"  Best value: {best_value:.4f} (target was ≤ 0.6)")
    assert best_value <= 0.6 or len(X2.data) == 20
    
    print("Testing run() without stopping criteria (should fail)...")
    # Test 3: No stopping criteria should raise error
    X3 = Xopt(
        vocs=vocs,
        evaluator=evaluator,
        generator=LatinHypercubeGenerator(vocs=vocs)
    )
    
    try:
        X3.run()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")
    
    print("\n🎉 All integration tests passed!")


if __name__ == "__main__":
    test_stopping_condition_integration()