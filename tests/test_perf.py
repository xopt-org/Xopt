import copy
import time

import numpy as np

from xopt import Xopt
from xopt.generators.random import RandomGenerator
from xopt.resources.testing import TEST_VOCS_BASE

NRUNS = 1000


class TestPerformance:
    def test_xopt_overhead(self):
        """Use random generator and compare time to direct calls"""

        # 15D is typical
        VOCS_15D = copy.deepcopy(TEST_VOCS_BASE)
        for i in range(2, 15):
            VOCS_15D.variables[f"x{i}"] = [0, 1]

        gen = RandomGenerator(vocs=VOCS_15D)

        config = {
            "generator": {
                "name": "random",
            },
            "evaluator": {"function": "xopt.resources.testing.xtest_callable"},
            "vocs": VOCS_15D,
        }
        X = Xopt.from_dict(config)

        gen.generate(1)
        t1 = time.perf_counter()
        times = np.zeros(NRUNS)
        for i in range(NRUNS):
            gen.generate(1)
            times[i] = time.perf_counter()
        times -= t1
        tpure = times[-1]

        X.random_evaluate(1)
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
        print(
            f"Xopt ratio: {txopt / tpure} ({(txopt-tpure)/NRUNS * 1e3:.3f}ms per "
            f"call)"
        )
        print(f"Ratio every 100 steps: {ratio_array[::100]}")

        if txopt / tpure > 100:
            raise ValueError("Xopt overhead is too large")
