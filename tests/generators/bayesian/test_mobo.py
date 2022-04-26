from copy import deepcopy

import torch
from botorch.sampling import SobolQMCNormalSampler

from xopt.generators.bayesian import MOBOGenerator
from xopt.resources.testing import TEST_VOCS_DATA, TEST_VOCS_BASE, xtest_callable
from xopt import Xopt, Evaluator


class TestBayesianExplorationGenerator:
    def test_init(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.objectives.update({"y2": "MINIMIZE"})
        gen = MOBOGenerator(vocs)

        print(f"\n{gen.options.dict()}")
