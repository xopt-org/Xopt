import pytest

from xopt.generators.bayesian.options import AcquisitionOptions, OptimizationOptions


class TestBayesianOptions:
    def test_default_serialization(self):
        for ele in [AcquisitionOptions(), OptimizationOptions()]:
            ele.json()

    def test_num_restarts(self):
        with pytest.raises(ValueError):
            OptimizationOptions(num_restarts=25)
