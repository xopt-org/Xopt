import copy
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest
from scipy.optimize import minimize

from xopt import Xopt
from xopt.errors import SeqGeneratorError
from xopt.generators.sequential.scipy import (
    BOUNDED_METHODS,
    ScipyGenerator,
    _StopSession,
)
from xopt.vocs import VOCS


def sphere(input_dict):
    return {"y": float(sum(v**2 for v in input_dict.values()))}


def _direct_scipy_sequence(vocs: VOCS, method: str, maxiter: int = 30):
    mins, maxs = np.array(vocs.bounds).T
    bounds = list(zip(mins, maxs))
    x0 = np.array([1.7, -1.3], dtype=float)
    cache = {}
    sequence = []

    def objective(x):
        key = tuple(np.round(np.array(x, dtype=float), decimals=12))
        if key in cache:
            return cache[key]

        point = np.array(x, dtype=float)
        sequence.append(point)
        y_value = sphere(dict(zip(vocs.variable_names, point.tolist())))["y"]
        cache[key] = y_value
        return y_value

    minimize(
        objective,
        x0,
        method=method,
        bounds=bounds,
        tol=1e-8,
        options={"maxiter": maxiter},
    )

    return sequence


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

    def test_scipy_generator_model_dump_roundtrip_continuation(self):
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        gen = ScipyGenerator(
            vocs=vocs,
            method="Powell",
            initial_point={"x0": 1.2, "x1": -1.1},
            tol=1e-8,
        )

        for _ in range(5):
            candidate = gen.generate(1)[0]
            y = sphere(candidate)["y"]
            gen.add_data(pd.DataFrame([{**candidate, "y": y}]))

        restored: ScipyGenerator = ScipyGenerator.model_validate(gen.model_dump())
        restored.set_data(gen.data.copy(deep=True))

        reference = ScipyGenerator(
            vocs=vocs,
            method="Powell",
            initial_point={"x0": 1.2, "x1": -1.1},
            tol=1e-8,
        )
        reference.set_data(gen.data.copy(deep=True))

        restored_candidate = restored.generate(1)[0]
        reference_candidate = reference.generate(1)[0]
        restored_x = np.array(
            [restored_candidate[name] for name in vocs.variable_names]
        )
        reference_x = np.array(
            [reference_candidate[name] for name in vocs.variable_names]
        )

        np.testing.assert_allclose(restored_x, reference_x, rtol=0.0, atol=1e-12)

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

    @pytest.mark.parametrize("method", BOUNDED_METHODS)
    def test_selected_points_match_scipy_minimize(self, method):
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        expected_sequence = _direct_scipy_sequence(vocs, method, maxiter=30)
        assert len(expected_sequence) > 0

        gen = ScipyGenerator(
            vocs=vocs,
            method=method,
            initial_point={"x0": 1.7, "x1": -1.3},
            tol=1e-8,
            options={"maxiter": 30},
        )

        for expected_x in expected_sequence:
            candidate = gen.generate(1)[0]
            actual_x = np.array([candidate[name] for name in vocs.variable_names])

            np.testing.assert_allclose(actual_x, expected_x, rtol=0.0, atol=1e-12)

            y = sphere(candidate)["y"]
            gen.add_data(pd.DataFrame([{**candidate, "y": y}]))

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

    def test_validate_method_rejects_whitespace_string(self):
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        with pytest.raises(
            ValueError,
            match="scipy method '' is not supported; choose one of .*",
        ):
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

    def test_deepcopy_with_data_copies_outcome_and_cache(self):
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        gen = ScipyGenerator(vocs=vocs, method="Powell")
        data = pd.DataFrame(
            [{"x0": 0.2, "x1": -0.3, "y": 0.13}, {"x0": -0.1, "x1": 0.4, "y": 0.17}]
        )
        gen._set_data(data)

        copied = copy.deepcopy(gen)

        assert copied is not gen
        assert copied.data is not gen.data
        assert copied._last_outcome == pytest.approx(gen._last_outcome)
        assert copied._cache == copied._build_cache()

    def test_state_roundtrip_rebuilds_runtime_state_and_last_outcome(self):
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        gen = ScipyGenerator(vocs=vocs, method="Powell")
        data = pd.DataFrame(
            [{"x0": 0.2, "x1": -0.3, "y": 0.13}, {"x0": -0.1, "x1": 0.4, "y": 0.17}]
        )
        gen._set_data(data)

        state = gen.__getstate__()
        restored = ScipyGenerator.model_validate(gen.model_dump())
        restored.__setstate__(state)

        assert restored._cache == restored._build_cache()
        assert restored._last_outcome == pytest.approx(0.17)

    def test_raise_session_error_maps_unknown_solver(self):
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        gen = ScipyGenerator(vocs=vocs, method="Powell")
        gen._session_exception = ValueError("Unknown solver foo")

        with pytest.raises(
            RuntimeError,
            match="scipy method 'Powell' is not available in this environment",
        ):
            gen._raise_session_error_if_present()

    def test_raise_session_error_maps_cannot_handle_bounds(self):
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        gen = ScipyGenerator(vocs=vocs, method="Powell")
        gen._session_exception = ValueError("Method Powell cannot handle bounds")

        with pytest.raises(
            RuntimeError,
            match="does not support bounds",
        ):
            gen._raise_session_error_if_present()

    def test_objective_raises_stop_session_when_stop_event_set(self):
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        gen = ScipyGenerator(vocs=vocs, method="Powell")
        gen._stop_event.set()

        with pytest.raises(_StopSession):
            gen._objective(np.array([0.0, 0.0]))

    def test_add_data_empty_dataframe(self):
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        gen = ScipyGenerator(vocs=vocs, method="Powell")
        # _add_data with empty dataframe should return early without error
        gen._add_data(pd.DataFrame())
        assert gen._last_outcome is None

    def test_unknown_solver_raises_validation_error(self):
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
        with pytest.raises(
            ValidationError,
            match="scipy method .* is not supported; choose one of .*",
        ):
            Xopt.from_yaml(YAML)

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

    def test_convergence_path_without_data_raises_runtime_error(self):
        """Convergence without queued requests and without data raises an error."""
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        gen = ScipyGenerator(
            vocs=vocs, method="Powell", initial_point={"x0": 0.5, "x1": -0.5}
        )

        with patch("xopt.generators.sequential.scipy.minimize", return_value=None):
            with pytest.raises(
                RuntimeError, match="scipy minimize converged without available data"
            ):
                gen._generate()

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
