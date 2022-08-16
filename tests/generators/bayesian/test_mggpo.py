from copy import deepcopy

import numpy as np
import pytest
import yaml

from xopt.evaluator import Evaluator
from xopt.base import Xopt
from xopt.generators.bayesian.mggpo import MGGPOGenerator
from xopt.resources.test_functions.tnk import evaluate_TNK, tnk_vocs
from xopt.resources.testing import TEST_VOCS_BASE


class TestMGPO:
    def test_init(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.objectives.update({"y2": "MINIMIZE"})
        MGGPOGenerator(vocs)

    def test_script(self):
        evaluator = Evaluator(function=evaluate_TNK)

        # test check options
        vocs = deepcopy(tnk_vocs)
        gen = MGGPOGenerator(vocs)
        X = Xopt(evaluator=evaluator, generator=gen, vocs=vocs)
        X.step()
        X.generator.generate(10)
