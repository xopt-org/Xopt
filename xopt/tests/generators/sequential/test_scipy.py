import json
from unittest.mock import patch

import pandas as pd
import pytest

from xopt import Xopt
from xopt.errors import SeqGeneratorError
from xopt.generators.sequential.scipy import ScipyGenerator
from xopt.vocs import VOCS


def sphere(input_dict):
    return {"y": float(sum(v**2 for v in input_dict.values()))}


class TestScipyGenerator:
    def test_scipy_generate_single_point(self):
        YAML = """
        generator:
            name: scipy
            method: Powell
            initial_point: {x0: 0.5, x1: -0.5}
            vocs:
                variables:
                    x0: [-5, 5]
                    x1: [-5, 5]
                objectives: {y: MINIMIZE}
        evaluator:
            function: xopt.tests.generators.sequential.test_scipy.sphere
        """
        X = Xopt.from_yaml(YAML)
        gen: ScipyGenerator = X.generator

        first = gen.generate(1)
        assert len(first) == 1
        assert set(first[0].keys()) == {"x0", "x1"}

        # test without initial point
        YAML_NO_INIT = """
        generator:
            name: scipy
            method: Powell
            vocs:
                variables:
                    x0: [-5, 5]
                    x1: [-5, 5]
                objectives: {y: MINIMIZE}
        evaluator:
            function: xopt.tests.generators.sequential.test_scipy.sphere
        """
        X_no_init = Xopt.from_yaml(YAML_NO_INIT)
        X_no_init.random_evaluate(
            1
        )  # generate some data to build an initial point from
        X_no_init.step()

    def test_scipy_generate_multiple_points(self):
        YAML = """
        generator:
            name: scipy
            method: Powell
            initial_point: {x0: 0.5, x1: -0.5}
            vocs:
                variables:
                    x0: [-5, 5]
                    x1: [-5, 5]
                objectives: {y: MINIMIZE}
        evaluator:
            function: xopt.tests.generators.sequential.test_scipy.sphere
        """
        X = Xopt.from_yaml(YAML)
        with pytest.raises(SeqGeneratorError):
            X.generator.generate(2)

    def test_scipy_generate_and_restart(self):
        YAML = """
        generator:
            name: scipy
            method: Powell
            initial_point: {x0: 1.2, x1: -1.1}
            options:
                maxiter: 200
            vocs:
                variables:
                    x0: [-5, 5]
                    x1: [-5, 5]
                objectives: {y: MINIMIZE}
        evaluator:
            function: xopt.tests.generators.sequential.test_scipy.sphere
        """
        X = Xopt.from_yaml(YAML)

        for _ in range(8):
            X.step()

        assert len(X.data) == 8

        state = X.json()
        X2 = Xopt.model_validate(json.loads(state))
        X2.step()

        assert len(X2.data) == 9

    def test_scipy_generator_direct(self):
        YAML = """
        generator:
            name: scipy
            method: Powell
            initial_point: {x0: 1.0, x1: 1.0}
            vocs:
                variables:
                    x0: [-5, 5]
                    x1: [-5, 5]
                objectives: {y: MINIMIZE}
        evaluator:
            function: xopt.tests.generators.sequential.test_scipy.sphere
        """
        X = Xopt.from_yaml(YAML)
        gen: ScipyGenerator = X.generator

        first = gen.generate(1)
        assert len(first) == 1
        assert set(first[0].keys()) == {"x0", "x1"}

    def test_scipy_reset(self):
        YAML = """
        generator:
            name: scipy
            method: Powell
            initial_point: {x0: 0.75, x1: -0.25}
            vocs:
                variables:
                    x0: [-5, 5]
                    x1: [-5, 5]
                objectives: {y: MINIMIZE}
        evaluator:
            function: xopt.tests.generators.sequential.test_scipy.sphere
        """
        X = Xopt.from_yaml(YAML)
        gen: ScipyGenerator = X.generator

        X.step()
        assert gen.is_active
        assert gen._last_candidate is not None
        assert gen._last_outcome is not None

        gen.reset()

        assert not gen.is_active
        assert gen._last_candidate is None
        assert gen._last_outcome is None

        candidate = gen.generate(1)
        assert len(candidate) == 1
        assert set(candidate[0].keys()) == {"x0", "x1"}

    def test_scipy_sequence_repeats_after_reset(self):
        YAML = """
        generator:
            name: scipy
            method: Powell
            initial_point: {x0: 0.75, x1: -0.25}
            options:
                maxiter: 200
            vocs:
                variables:
                    x0: [-5, 5]
                    x1: [-5, 5]
                objectives: {y: MINIMIZE}
        evaluator:
            function: xopt.tests.generators.sequential.test_scipy.sphere
        """
        X = Xopt.from_yaml(YAML)
        gen: ScipyGenerator = X.generator

        def run_sequence(n_steps: int):
            seq = []
            for _ in range(n_steps):
                candidate = gen.generate(1)[0]
                seq.append((candidate["x0"], candidate["x1"]))

                objective = sphere(candidate)["y"]
                gen.add_data(
                    pd.DataFrame(
                        [
                            {
                                "x0": candidate["x0"],
                                "x1": candidate["x1"],
                                "y": objective,
                            }
                        ]
                    )
                )
            return seq

        first_sequence = run_sequence(5)
        gen.reset()
        second_sequence = run_sequence(5)

        assert second_sequence == pytest.approx(first_sequence)

    def test_validate_method_rejects_whitespace_string(self):
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        with pytest.raises(ValueError, match="method cannot be empty"):
            ScipyGenerator(vocs=vocs, method="   ")

    def test_validate_initial_point_empty_dict(self):
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        with pytest.raises(
            ValueError, match="initial_point cannot be an empty dictionary"
        ):
            ScipyGenerator(vocs=vocs, initial_point={})

    def test_add_data_empty_dataframe(self):
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        gen = ScipyGenerator(vocs=vocs, method="Powell")
        # _add_data with empty dataframe should return early without error
        gen._add_data(pd.DataFrame())
        assert gen._last_outcome is None

    def test_unknown_solver_raises_runtime_error(self):
        YAML = """
        generator:
            name: scipy
            method: NOT_A_REAL_METHOD
            initial_point: {x0: 0.5, x1: -0.5}
            vocs:
                variables:
                    x0: [-5, 5]
                    x1: [-5, 5]
                objectives: {y: MINIMIZE}
        evaluator:
            function: xopt.tests.generators.sequential.test_scipy.sphere
        """
        X = Xopt.from_yaml(YAML)
        with pytest.raises(RuntimeError, match="scipy method .* is not available"):
            X.generator.generate(1)

    def test_non_solver_value_error_reraises(self):
        """A ValueError that doesn't mention 'unknown solver' should propagate."""
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        gen = ScipyGenerator(
            vocs=vocs, method="Powell", initial_point={"x0": 0.5, "x1": -0.5}
        )

        def _bad_minimize(*args, **kwargs):
            raise ValueError("some other value error")

        with patch("xopt.generators.sequential.scipy.minimize", _bad_minimize):
            with pytest.raises(ValueError, match="some other value error"):
                gen._generate()

    def test_convergence_path_returns_last_point(self):
        """When scipy converges using only cached data, the last cached point is returned."""
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        gen = ScipyGenerator(
            vocs=vocs, method="Powell", initial_point={"x0": 0.5, "x1": -0.5}
        )
        # Populate data so the generator has something to return
        data = pd.DataFrame([{"x0": 0.1, "x1": 0.2, "y": 0.05}])
        gen._set_data(data)

        # Mock minimize to complete without calling the objective (simulates convergence)
        with patch("xopt.generators.sequential.scipy.minimize", return_value=None):
            result = gen._generate()

        assert result is not None
        assert len(result) == 1
        assert result[0]["x0"] == pytest.approx(0.1)
        assert result[0]["x1"] == pytest.approx(0.2)

    def test_convergence_path_with_constants(self):
        """Convergence path includes constants in returned point."""
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
            constants={"c": 3.14},
        )
        gen = ScipyGenerator(
            vocs=vocs, method="Powell", initial_point={"x0": 0.5, "x1": -0.5}
        )
        data = pd.DataFrame([{"x0": 0.1, "x1": 0.2, "y": 0.05, "c": 3.14}])
        gen._set_data(data)

        with patch("xopt.generators.sequential.scipy.minimize", return_value=None):
            result = gen._generate()

        assert result is not None
        assert "c" in result[0]

    def test_generate_with_constants(self):
        """Constants are included in generated candidates."""
        YAML = """
        generator:
            name: scipy
            method: Powell
            initial_point: {x0: 0.5, x1: -0.5}
            vocs:
                variables:
                    x0: [-5, 5]
                    x1: [-5, 5]
                objectives: {y: MINIMIZE}
                constants: {c: 1.0}
        evaluator:
            function: xopt.tests.generators.sequential.test_scipy.sphere
        """
        X = Xopt.from_yaml(YAML)
        candidate = X.generator.generate(1)
        assert len(candidate) == 1
        assert "c" in candidate[0]
