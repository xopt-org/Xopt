import numpy as np
import pytest
from xopt import VOCS


def test_ackley_evaluation():
    from xopt.resources.test_functions.ackley_20 import (
        evaluate_ackley_np,
        evaluate_ackley,
    )

    x = {f"x{i}": 0.0 for i in range(20)}
    evaluate_ackley_np(x)
    evaluate_ackley(x)


def test_haverly_pooling():
    from xopt.resources.test_functions.haverly_pooling import (
        evaluate_haverly,
    )

    x = {f"x{i}": 50.0 for i in range(1, 10)}
    evaluate_haverly(x)


def test_modified_tnk():
    from xopt.resources.test_functions.modified_tnk import (
        evaluate_modified_TNK,
    )

    x = {"x1": 1.0, "x2": 1.0}
    evaluate_modified_TNK(x)


def test_rosenbrock_evaluation():
    from xopt.resources.test_functions.rosenbrock import (
        evaluate_rosenbrock,
        make_rosenbrock_vocs,
    )

    x = {f"x{i}": 1.0 for i in range(5)}
    evaluate_rosenbrock(x)

    vocs = make_rosenbrock_vocs(5)
    assert isinstance(vocs, VOCS)
    assert len(vocs.variable_names) == 5


def test_sinusoid_1d():
    from xopt.resources.test_functions.sinusoid_1d import (
        evaluate_sinusoid,
        sinusoid_vocs,
    )

    x = {"x1": 1.0}
    evaluate_sinusoid(x)

    assert isinstance(sinusoid_vocs, VOCS)
    assert len(sinusoid_vocs.variable_names) == 1


def test_tnk():
    from xopt.resources.test_functions.tnk import evaluate_TNK

    x = {"x1": 0.5, "x2": 0.5}
    evaluate_TNK(x)

    # test with raised ValueError
    with pytest.raises(ValueError):
        evaluate_TNK(inputs=x, raise_probability=1.0)


@pytest.mark.parametrize("problem_index", [1, 2, 3])
def test_zdt(problem_index):
    from xopt.resources.test_functions.zdt import construct_zdt

    vocs, evaluate, reference_point = construct_zdt(5, problem_index=problem_index)
    x = {f"x{i}": 0.5 for i in range(1, 6)}
    evaluate(x)

    assert isinstance(vocs, VOCS)
    assert len(vocs.variable_names) == 5
    assert isinstance(reference_point, dict)

    with pytest.raises(NotImplementedError):
        construct_zdt(5, problem_index=4)


def test_multi_objective_problems():
    from xopt.resources.test_functions.multi_objective import (
        DTLZ2,
        LinearMO,
        QuadraticMO,
    )

    for ele in [DTLZ2(), LinearMO(), QuadraticMO()]:
        assert isinstance(ele.ref_point, np.ndarray)
        assert isinstance(ele.ref_point_dict, dict)
        assert isinstance(ele.vocs, VOCS)
        assert isinstance(ele.VOCS, VOCS)
        assert isinstance(
            ele.evaluate_dict({f"x{i + 1}": 0.0 for i in range(ele.n_var)}), dict
        )
        assert isinstance(ele.bounds, list)
        assert isinstance(ele.bounds_numpy, np.ndarray)
        assert isinstance(ele.optimal_value, type(None))

    DTLZ2()._max_hv
