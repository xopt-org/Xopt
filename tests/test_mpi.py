import os

import pytest
import yaml

havempi = False
try:
    from mpi4py import MPI  # noqa: F401

    havempi = True
except ImportError:
    pass

needsmpi = pytest.mark.skipif(not havempi, reason="MPI not available")


class TestMPI:
    @needsmpi
    def test_mpi(self):
        from xopt.mpi.run import run_mpi

        YAML = """
                max_evaluations: 5
                evaluator:
                    function: xopt.resources.test_functions.tnk.evaluate_TNK
                    function_kwargs:
                        a: 999
                    max_workers: 2

                generator:
                    name: random

                vocs:
                    variables:
                        x1: [0, 3.14159]
                        x2: [0, 3.14159]
                    objectives: {y1: MINIMIZE, y2: MINIMIZE}
                    constraints:
                        c1: [GREATER_THAN, 0]
                        c2: [LESS_THAN, 0.5]
                    constants: {a: dummy_constant}

                """

        # run batched mode
        run_mpi(yaml.safe_load(YAML), 0, False, None)

        # run asynch mode
        run_mpi(yaml.safe_load(YAML), 0, True, None)

        # test with file
        with open("test.yml", "w") as f:
            yaml.dump(yaml.safe_load(YAML), f)

        run_mpi(yaml.safe_load(open("test.yml")), 0, False, None)

    @needsmpi
    def test_with_cnsga(self):
        from xopt.mpi.run import run_mpi

        YAML = """
        max_evaluations: 10
        generator:
            name: cnsga
            population_size: 64

        evaluator:
            function: xopt.resources.test_functions.tnk.evaluate_TNK
            function_kwargs:
                sleep: 0
                random_sleep: 0.1

        vocs:
            variables:
                x1: [0, 3.14159]
                x2: [0, 3.14159]
            objectives: {y1: MINIMIZE, y2: MINIMIZE}
            constraints:
                c1: [GREATER_THAN, 0]
                c2: [LESS_THAN, 0.5]
            constants: {a: dummy_constant}
        """

        # run batched mode
        run_mpi(yaml.safe_load(YAML), 0, False, None)

        # run asynch mode
        run_mpi(yaml.safe_load(YAML), 0, True, None)

    @pytest.fixture(scope="module", autouse=True)
    def clean_up(self):
        yield
        files = ["test.yml"]
        for f in files:
            if os.path.exists(f):
                os.remove(f)
