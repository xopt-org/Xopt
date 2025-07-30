import pytest
import pandas as pd
import numpy as np

from xopt.errors import SeqGeneratorError
from xopt.generators.sequential.sequential_generator import SequentialGenerator
from xopt.vocs import VOCS


class TestSequentialGenerator(SequentialGenerator):
    __test__ = False
    supports_single_objective: bool = True

    def _add_data(self, new_data: pd.DataFrame):
        pass

    def _generate(self, first_gen: bool = False) -> dict:
        return {"x1": 0.5, "x2": 0.5}

    def _reset(self):
        pass


@pytest.fixture
def sample_vocs():
    return VOCS(
        variables={"x1": [0.0, 1.0], "x2": [0.0, 1.0]},
        constraints={},
        objectives={"y1": "MINIMIZE"},
        constants={},
        observables=[],
    )


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {"x1": [0.1, 0.2, 0.3], "x2": [0.4, 0.5, 0.6], "y1": [1.0, 0.8, 0.6]}
    )


def test_add_data(sample_vocs, sample_data):
    gen = TestSequentialGenerator(vocs=sample_vocs)
    gen.add_data(sample_data)
    assert gen.data.equals(sample_data)

    gen.is_active = True
    gen._last_candidate = [{"x1": 0.3, "x2": 0.6}]
    with pytest.raises(SeqGeneratorError):
        gen.add_data(sample_data)


def test_generate(sample_vocs):
    gen = TestSequentialGenerator(vocs=sample_vocs)
    candidate = gen.generate()
    assert candidate == {"x1": 0.5, "x2": 0.5}
    assert gen._last_candidate == candidate

    with pytest.raises(SeqGeneratorError):
        gen.generate(n_candidates=2)


def test_reset(sample_vocs):
    gen = TestSequentialGenerator(vocs=sample_vocs)
    gen.is_active = True
    gen._last_candidate = {"x1": 0.5, "x2": 0.5}
    gen.reset()
    assert not gen.is_active
    assert gen._last_candidate is None


def test_get_initial_point(sample_vocs, sample_data):
    gen = TestSequentialGenerator(vocs=sample_vocs)
    gen.data = sample_data
    x0, f0 = gen._get_initial_point()
    assert np.allclose(x0, [0.3, 0.6])
    assert np.allclose(f0, [0.6])

    gen.data = None
    with pytest.raises(ValueError):
        gen._get_initial_point()
