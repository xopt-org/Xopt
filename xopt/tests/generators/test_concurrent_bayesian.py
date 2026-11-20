"""Concurrent local Bayesian optimization regression tests.

Exercise independent Xopt applications sharing a PyTorch runtime.

CUDA cases also exercise the automatic first-use initialization
workaround for PyTorch issue #90613. No explicit CUDA warmup
is performed by this test.
"""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pandas as pd
import pytest
import torch

from gest_api.vocs import VOCS

from xopt import Evaluator, Xopt
from xopt.generators.bayesian.upper_confidence_bound import (
    UpperConfidenceBoundGenerator,
)
from xopt.generators.bayesian.expected_improvement import (
    ExpectedImprovementGenerator,
)
from xopt.generators.bayesian.bax_generator import BaxGenerator
from xopt.generators.bayesian.bax.algorithms import GridOptimize


@pytest.mark.parametrize("mode", ["cpu", "cuda", "mixed"])
def test_concurrent_local_bayesian(mode):
    """Run six independent Xopt applications concurrently."""

    if mode != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    workers = 6
    steps = 5
    barrier = Barrier(workers)

    def build_application(index):
        algorithm = ["ucb", "bax", "ei"][index % 3]
        output = f"y_{index}"
        target = (index + 1) / 7.0

        use_cuda = mode == "cuda" or (mode == "mixed" and index >= 3)

        def evaluate(inputs):
            x = float(inputs["x"])
            return {output: (x - target) ** 2 + index}

        if algorithm == "bax":
            vocs = VOCS(
                variables={"x": [0.0, 1.0]},
                observables=[output],
            )

            generator = BaxGenerator(
                vocs=vocs,
                use_cuda=use_cuda,
                algorithm=GridOptimize(
                    observable_names_ordered=[output],
                    n_mesh_points=20,
                ),
            )
        else:
            vocs = VOCS(
                variables={"x": [0.0, 1.0]},
                objectives={output: "MINIMIZE"},
            )

            generator_class = (
                UpperConfidenceBoundGenerator
                if algorithm == "ucb"
                else ExpectedImprovementGenerator
            )

            generator = generator_class(
                vocs=vocs,
                use_cuda=use_cuda,
            )

        application = Xopt(
            generator=generator,
            evaluator=Evaluator(function=evaluate),
        )

        return (
            application,
            algorithm,
            output,
            target,
            use_cuda,
        )

    applications = [build_application(i) for i in range(workers)]

    def run_application(index):
        X, algorithm, output, target, use_cuda = applications[index]

        X.evaluate_data(pd.DataFrame({"x": [0.1, 0.3, 0.6, 0.9]}))

        # Synchronize first model training across all six
        # workers to expose CUDA first-use races.
        barrier.wait(timeout=60)

        for _ in range(steps):
            X.step()

        assert X.n_data == 4 + steps

        x = X.data["x"].to_numpy()
        y = X.data[output].to_numpy()

        assert np.all((x >= 0.0) & (x <= 1.0))

        np.testing.assert_allclose(
            y,
            (x - target) ** 2 + index,
            rtol=1e-10,
            atol=1e-12,
        )

        for other in range(workers):
            if other != index:
                assert f"y_{other}" not in X.data.columns

        assert X.generator.model is not None

        devices = {
            parameter.device.type for parameter in X.generator.model.parameters()
        }

        expected = "cuda" if use_cuda else "cpu"
        assert devices == {expected}

        print(
            f"Xopt {index + 1}: "
            f"algorithm={algorithm.upper()}, "
            f"device={expected}, "
            f"observations={X.n_data}, PASS"
        )

    print(f"\nExecution mode: {mode}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Intra-op threads: {torch.get_num_threads()}")
    print(f"Inter-op threads: {torch.get_num_interop_threads()}")
    print(f"CUDA initialized before: {torch.cuda.is_initialized()}")

    cuda_was_initialized = torch.cuda.is_initialized()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_application, i) for i in range(workers)]

        for future in futures:
            future.result(timeout=300)

    if mode == "cpu" and not cuda_was_initialized:
        assert not torch.cuda.is_initialized()

    print(f"CUDA initialized after: {torch.cuda.is_initialized()}")
    print(f"Total observations: {workers * (4 + steps)}")
    print("ALL SIX XOPT APPLICATIONS PASSED")
