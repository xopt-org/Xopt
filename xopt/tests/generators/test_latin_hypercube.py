from copy import deepcopy

import numpy as np
from scipy.stats import qmc

from xopt.generators.scipy.latin_hypercube import LatinHypercubeGenerator
from xopt.resources.testing import TEST_VOCS_BASE


class TestLatinHypercubeGenerator:
    def test_n_sample(self):
        # Create the generator and test name
        gen = LatinHypercubeGenerator(vocs=TEST_VOCS_BASE, batch_size=128)
        assert gen.name == "latin_hypercube"

        # Try to get samples and confirm that batching works correctly
        for _ in range(32):
            assert len(gen.generate(53)) == 53

    def test_yaml(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives = {"y": "MAXIMIZE", "z": "MINIMIZE"}
        gen = LatinHypercubeGenerator(vocs=test_vocs)
        gen.yaml()

    def test_scipy_comparison(self, n_samples=128, max_dim_power_two=3):
        configs = [
            {"scramble": False, "optimization": None},
            {"scramble": True, "optimization": None},
            {"scramble": True, "optimization": "random-cd"},
        ]
        for dim in 2 ** np.arange(1, max_dim_power_two + 1):
            for config in configs:
                # Get the samples from xopt
                test_vocs = deepcopy(TEST_VOCS_BASE)
                test_vocs.variables = {f"x{i+1}": [0, 1] for i in range(dim)}
                gen = LatinHypercubeGenerator(
                    vocs=test_vocs, seed=1, batch_size=n_samples, **config
                )
                samps_xopt = np.array(
                    [
                        [x[k] for k in test_vocs.variable_names]
                        for x in gen.generate(n_samples)
                    ]
                )

                # Get the samples from scipy
                sampler = qmc.LatinHypercube(d=dim, seed=1, **config)
                samps_scipy = sampler.random(n=n_samples)

                # Compare
                np.testing.assert_allclose(samps_scipy, samps_xopt)
