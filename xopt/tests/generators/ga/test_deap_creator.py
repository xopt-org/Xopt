import array
import warnings
import copy
import pytest

import xopt.generators.ga.deap_creator as deap_creator


def test_array_deepcopy_and_reduce():
    deap_creator._array.typecode = "d"
    arr = deap_creator._array([1.0, 2.0, 3.0])
    arr.foo = "bar"
    arr2 = copy.deepcopy(arr)
    assert arr2 == arr
    assert arr2.foo == "bar"
    # Test __reduce__
    reduced = arr.__reduce__()
    assert reduced[0] is arr.__class__
    assert list(reduced[1][0]) == [1.0, 2.0, 3.0]
    assert isinstance(reduced[2], dict)


def test_meta_create_and_create():
    # Create a new class with a static and instance attribute
    deap_creator.create("Foo", list, bar=dict, spam=1)
    Foo = getattr(deap_creator, "Foo")
    f = Foo([1, 2, 3])
    assert isinstance(f, list)
    assert hasattr(f, "bar")
    assert isinstance(f.bar, dict)
    assert Foo.spam == 1
    # Test warning for duplicate class name
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        deap_creator.create("Foo", list, bar=dict, spam=2)
        assert any("already been created" in str(wi.message) for wi in w)


def test_class_replacers():
    # Should contain array.array
    assert array.array in deap_creator.class_replacers
    assert issubclass(deap_creator.class_replacers[array.array], array.array)


def test_metacreator_reduce():
    # Create a class and check __reduce__
    deap_creator.create("Bar", list, baz=int)
    Bar = getattr(deap_creator, "Bar")
    bar_instance = Bar([1, 2, 3])
    reduced = bar_instance.__reduce__()
    assert isinstance(reduced, tuple)
    assert callable(reduced[0])
    # Find the data in the reduce tuple
    found = False
    for item in reduced[1]:
        if isinstance(item, (list, tuple)) and list(item) == [1, 2, 3]:
            found = True
            break
    assert found, f"Reduce tuple does not contain the expected data: {reduced}"


def test_numpy_array_replacer():
    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy not available")
    arr = deap_creator.class_replacers[np.ndarray]([1, 2, 3])
    arr.foo = "bar"
    arr2 = copy.deepcopy(arr)
    assert np.all(arr2 == arr)
    assert arr2.foo == "bar"
    # Test __reduce__
    reduced = arr.__reduce__()
    assert reduced[0] is arr.__class__
    assert reduced[1][0] == [1, 2, 3]
    assert isinstance(reduced[2], dict)
