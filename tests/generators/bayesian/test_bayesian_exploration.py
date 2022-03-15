import torch
from botorch.sampling import SobolQMCNormalSampler

from xopt.generators.bayesian import BayesianExplorationGenerator
from xopt.resources.testing import TEST_VOCS_DATA, TEST_VOCS_BASE, test_callable
from xopt import XoptBase, Evaluator


class TestBayesianExplorationGenerator:
    def test_init(self):
        gen = BayesianExplorationGenerator(TEST_VOCS_BASE)

    def test_generate(self):
        gen = BayesianExplorationGenerator(TEST_VOCS_BASE)

        sampler = SobolQMCNormalSampler(num_samples=1)
        gen.options["acqf_kw"].update({"sampler": sampler})
        gen.options["optim_kw"].update({"num_restarts": 1, "raw_samples": 1})
        candidate = gen.generate(TEST_VOCS_DATA, 5)
        assert len(candidate) == 5

    def test_in_xopt(self):
        evaluator = Evaluator(test_callable)
        generator = BayesianExplorationGenerator(TEST_VOCS_BASE)

        xopt = XoptBase(generator, evaluator, TEST_VOCS_BASE)

        # initialize with single initial candidate
        xopt.step()

        # now use bayes opt
        for _ in range(1):
            xopt.step()
