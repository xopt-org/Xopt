import time

import numpy as np

from xopt import Xopt
from xopt.generators.random import RandomGenerator
from xopt.resources.testing import TEST_VOCS_BASE

NRUNS = 5000


class TestPerformance:
    def test_xopt_overhead(self):
        """ Use random generator and compare time to direct calls """
        gen = RandomGenerator(vocs=TEST_VOCS_BASE)

        config = {
            "generator": {
                "name": "random",
            },
            "evaluator": {"function": f"xopt.resources.testing.xtest_callable"},
            "vocs": TEST_VOCS_BASE,
        }
        X = Xopt.from_dict(config)

        t1 = time.perf_counter()
        times = np.zeros(NRUNS)
        for i in range(NRUNS):
            gen.generate(1)
            times[i] = time.perf_counter()
        times -= t1
        tpure = times[-1]

        t1 = time.perf_counter()
        times_xopt = np.zeros(NRUNS)
        for i in range(NRUNS):
            X.step()
            times_xopt[i] = time.perf_counter()
        times_xopt -= t1
        txopt = times_xopt[-1]

        ratio_array = times_xopt / times

        print(f"Pure generator time: {tpure}")
        print(f"Xopt time: {txopt}")
        print(f"Xopt overall overhead: {txopt / tpure}")

        print(f'Ratio every 100 steps: {ratio_array[::100]}')
