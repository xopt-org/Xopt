import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from xopt import VOCS
from xopt.generators.random import RandomGenerator

from xopt.resources import testing as testing_utils


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


class _SlotOnly:
    __slots__ = ("items",)

    def __init__(self, items):
        self.items = items


class _BranchContainer:
    def __init__(self):
        self.module = nn.Linear(2, 1)
        self.as_list = [pd.DataFrame({"x": [1.0]}), 1.0, "a"]
        self.as_dict = {"a": pd.DataFrame({"x": [2.0]})}
        self.as_set = {1, 2}
        # Re-visit the same object to exercise the visited-guard branch.
        self.shared = self.as_dict


class _FakeBayesGen:
    def __init__(self, data):
        self.data = data
        self._module = nn.Linear(2, 1)

    def _get_objective(self):
        return self._module

    def train_model(self, _data):
        return self._module

    def _get_sampler(self, _model):
        return self._module

    def get_acquisition(self, _model):
        return self._module


class _WarnGenerator:
    def generate(self, n):
        import warnings

        warnings.warn("runtime warning", RuntimeWarning)
        return [{"n": n}]


class _SilentGenerator:
    def generate(self, n):
        return [{"n": n}]


class _ContainsFalseDict(dict):
    def __contains__(self, _key):
        return False


def test_verify_state_device_matches_and_mismatch_raises():
    module = nn.Linear(2, 1)
    testing_utils.verify_state_device(module, torch.device("cpu"), "ok")

    with pytest.raises(ValueError, match="expected"):
        testing_utils.verify_state_device(module, torch.device("meta"), "bad")


def test_recursive_torch_device_scan_branches():
    visited = set()

    # Basic-type fast path.
    testing_utils.recursive_torch_device_scan(1.0, visited, torch.device("cpu"))

    # __slots__ path and nested list traversal.
    slot_obj = _SlotOnly([nn.Linear(2, 1), 3])
    testing_utils.recursive_torch_device_scan(slot_obj, visited, torch.device("cpu"))

    # Dict/list/set/module paths and visited short-circuit via shared object.
    container = _BranchContainer()
    testing_utils.recursive_torch_device_scan(
        container, visited, torch.device("cpu"), verbose=True
    )


def test_recursive_torch_device_scan_verbose_guard_paths():
    visited = set()

    # Exercise verbose path for basic types.
    testing_utils.recursive_torch_device_scan(
        1.0, visited, torch.device("cpu"), verbose=True
    )

    # Exercise verbose path for objects without __dict__.
    slot_obj = _SlotOnly([1])
    testing_utils.recursive_torch_device_scan(
        slot_obj, visited, torch.device("cpu"), verbose=True
    )

    # Exercise already-visited early-return branch.
    reused = object()
    visited.add(id(reused))
    testing_utils.recursive_torch_device_scan(
        reused, visited, torch.device("cpu"), verbose=True
    )


def test_check_generator_tensor_locations_with_and_without_data():
    empty = _FakeBayesGen(pd.DataFrame())
    testing_utils.check_generator_tensor_locations(empty, torch.device("cpu"))

    nonempty = _FakeBayesGen(pd.DataFrame({"x1": [0.1], "x2": [0.2]}))
    testing_utils.check_generator_tensor_locations(nonempty, torch.device("cpu"))


def test_check_dict_equal_paths():
    d1 = {"a": 1, "b": 2}
    d2 = {"a": 1, "b": 2}
    testing_utils.check_dict_equal(d1, d2)
    testing_utils.check_dict_equal(d1, {"a": 0, "b": 2}, excluded_keys=["a"])

    with pytest.raises(KeyError):
        testing_utils.check_dict_equal({"a": 1}, {"b": 1})

    with pytest.raises(ValueError):
        testing_utils.check_dict_equal({"a": 1}, {"a": 2})


def test_check_dict_equal_key_not_in_dict2_branch():
    dict1 = {"a": 1}
    dict2 = _ContainsFalseDict({"a": 1})

    with pytest.raises(KeyError):
        testing_utils.check_dict_equal(dict1, dict2)


def test_check_dict_allclose_paths():
    d1 = {
        "t": torch.tensor([1.0, 2.0]),
        "n": np.array([1.0, 2.0]),
        "s": "same",
        "f": 1.0,
    }
    d2 = {
        "t": torch.tensor([1.0, 2.0]),
        "n": np.array([1.0, 2.0]),
        "s": "same",
        "f": 1.0,
    }
    testing_utils.check_dict_allclose(d1, d2)
    testing_utils.check_dict_allclose(d1, {**d2, "s": "different"}, excluded_keys=["s"])

    with pytest.raises(KeyError):
        testing_utils.check_dict_allclose({"a": 1.0}, {"b": 1.0})

    with pytest.raises(ValueError):
        testing_utils.check_dict_allclose(
            {"t": torch.tensor([1.0])}, {"t": torch.tensor([2.0])}
        )

    with pytest.raises(ValueError):
        testing_utils.check_dict_allclose({"n": np.array([1.0])}, {"n": np.array([2.0])})

    with pytest.raises(ValueError):
        testing_utils.check_dict_allclose({"s": "a"}, {"s": "b"})


def test_reload_generator_helpers_and_generate_without_warnings():
    gen = RandomGenerator(vocs=testing_utils.TEST_VOCS_BASE)
    gen.add_data(testing_utils.TEST_VOCS_DATA.head(2).copy())

    gen_json = testing_utils.reload_gen_from_json(gen)
    gen_yaml = testing_utils.reload_gen_from_yaml(gen)

    assert isinstance(gen_json, RandomGenerator)
    assert isinstance(gen_yaml, RandomGenerator)
    assert gen_json.data.equals(gen.data)
    assert gen_yaml.data.equals(gen.data)

    with pytest.raises(AssertionError):
        testing_utils.reload_gen_from_json(object())
    with pytest.raises(AssertionError):
        testing_utils.reload_gen_from_yaml(object())

    candidates = testing_utils.generate_without_warnings(_SilentGenerator(), 2)
    assert candidates == [{"n": 2}]

    with pytest.raises(RuntimeError, match="Warnings"):
        testing_utils.generate_without_warnings(_WarnGenerator(), 1)


def test_create_set_options_helper_sets_attributes_and_optionally_data():
    data = testing_utils.TEST_VOCS_DATA.head(1).copy()
    setter = testing_utils.create_set_options_helper(
        data=data,
        n_restarts=7,
        n_monte_carlo_samples=11,
    )

    class _NumericalOptimizerWithIter:
        def __init__(self):
            self.n_restarts = 0
            self.max_iter = 99

    class _NumericalOptimizerNoIter:
        def __init__(self):
            self.n_restarts = 0

    class _Carrier:
        def __init__(self, numerical_optimizer):
            self.use_cuda = False
            self.numerical_optimizer = numerical_optimizer
            self.n_monte_carlo_samples = 0
            self.data = None

        def add_data(self, df):
            self.data = df

    gen_with_iter = _Carrier(_NumericalOptimizerWithIter())
    setter(gen_with_iter, use_cuda=True, add_data=True)
    assert gen_with_iter.use_cuda is True
    assert gen_with_iter.numerical_optimizer.n_restarts == 7
    assert gen_with_iter.numerical_optimizer.max_iter == 1
    assert gen_with_iter.n_monte_carlo_samples == 11
    assert gen_with_iter.data.equals(data)

    gen_no_iter = _Carrier(_NumericalOptimizerNoIter())
    setter(gen_no_iter, use_cuda=False, add_data=False)
    assert gen_no_iter.use_cuda is False
    assert gen_no_iter.numerical_optimizer.n_restarts == 7
    assert gen_no_iter.n_monte_carlo_samples == 11
    assert gen_no_iter.data is None
