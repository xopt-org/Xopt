import copy
import json
import pickle
import warnings
from threading import Event
from unittest.mock import patch

import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest
from scipy.optimize import OptimizeResult, minimize

from xopt import Evaluator, Xopt
from xopt.errors import SeqGeneratorError
from xopt.generators.sequential.scipy import (
    BOUNDED_METHODS,
    ScipyGenerator,
    _StopSession,
)
from xopt.vocs import VOCS


def sphere(input_dict):
    """Return the quadratic objective used by generator integration tests."""
    return {"y": float(sum(v**2 for v in input_dict.values()))}


def _direct_scipy_sequence(vocs: VOCS, method: str, maxiter: int = 30):
    """Collect uncached requests from scipy as the reference candidate sequence."""
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
        """Generate one point with either a configured start or existing history."""
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
        """Reject batch requests for a sequential optimizer."""
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
        """Resume Xopt evaluation after a full JSON serialization roundtrip."""
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
        """Match restored-generator candidates to a fresh replay of the same history."""
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

    @pytest.mark.parametrize("method", BOUNDED_METHODS)
    def test_selected_points_match_scipy_minimize(self, method):
        """Match every uncached candidate to direct scipy for each supported method."""
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
        """Clear active candidate/outcome state and allow generation after reset."""
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
        """Reject a solver name that becomes empty after whitespace normalization."""
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
        """Reject an explicitly supplied initial point without coordinates."""
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        with pytest.raises(
            ValueError, match="initial_point cannot be an empty dictionary"
        ):
            ScipyGenerator(vocs=vocs, initial_point={})

    @pytest.mark.parametrize(
        "roundtrip", [copy.deepcopy, lambda gen: pickle.loads(pickle.dumps(gen))]
    )
    def test_copy_with_data_preserves_outcome_and_cache(self, roundtrip):
        """Restore independent history, last outcome, cache, and idle runtime state."""
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        gen = ScipyGenerator(vocs=vocs, method="Powell")
        data = pd.DataFrame(
            [{"x0": 0.2, "x1": -0.3, "y": 0.13}, {"x0": -0.1, "x1": 0.4, "y": 0.17}]
        )
        gen._set_data(data)

        copied = roundtrip(gen)

        assert copied is not gen
        assert copied.data is not gen.data
        assert copied._last_outcome == pytest.approx(gen._last_outcome)
        assert copied._cache == copied._build_cache()
        assert copied._session_thread is None
        assert copied._cache_lock is not gen._cache_lock
        assert copied._request_queue is not gen._request_queue
        assert copied._response_queue is not gen._response_queue
        assert copied._stop_event is not gen._stop_event

    @pytest.mark.parametrize(
        "roundtrip", [copy.deepcopy, lambda gen: pickle.loads(pickle.dumps(gen))]
    )
    def test_copy_active_generator_preserves_sequential_state(self, roundtrip):
        """Copy a pending candidate without stopping or sharing the original worker."""
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        gen = ScipyGenerator(
            vocs=vocs,
            method="Powell",
            initial_point={"x0": 0.5, "x1": -0.5},
        )
        gen.set_data(pd.DataFrame(columns=["x0", "x1", "y"]))
        candidate = gen.generate(1)[0]
        session_thread = gen._session_thread
        assert session_thread is not None and session_thread.is_alive()

        copied = roundtrip(gen)
        assert gen._session_thread is session_thread
        assert session_thread.is_alive()
        assert copied.is_active
        assert copied._data_set
        assert copied._last_candidate == [candidate]

        copied.add_data(pd.DataFrame([{**candidate, "y": sphere(candidate)["y"]}]))
        assert len(copied.data) == 1
        assert copied._pending_request is None
        assert copied._response_queue.empty()
        requested_key = copied._point_key(np.array([candidate["x0"], candidate["x1"]]))
        assert copied._build_cache()[requested_key] == sphere(candidate)["y"]
        assert gen._pending_request == requested_key
        assert gen.data.empty

        copied.reset()
        gen.reset()

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            (
                "Unknown solver foo",
                "scipy method 'Powell' is not available in this environment",
            ),
            ("Method Powell cannot handle bounds", "does not support bounds"),
        ],
    )
    def test_raise_session_error_maps_solver_errors(self, message, expected):
        """Translate recognized worker ValueErrors into actionable runtime errors."""
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        gen = ScipyGenerator(vocs=vocs, method="Powell")
        gen._session_exception = ValueError(message)

        with pytest.raises(
            RuntimeError,
            match=expected,
        ):
            gen._raise_session_error_if_present()

    def test_objective_raises_stop_session_when_stop_event_set(self):
        """Reject objective calls belonging to an already stopped session."""
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        gen = ScipyGenerator(vocs=vocs, method="Powell")
        gen._stop_event.set()

        with pytest.raises(_StopSession):
            gen._objective(
                np.array([0.0, 0.0]),
                stop_event=gen._stop_event,
                request_queue=gen._request_queue,
                response_queue=gen._response_queue,
                cache=gen._cache,
                cache_lock=gen._cache_lock,
                stop_token=gen._stop_token,
            )

    def test_add_data_empty_dataframe(self):
        """Ignore an empty internal data update without changing the last outcome."""
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        gen = ScipyGenerator(vocs=vocs, method="Powell")
        gen._add_data(pd.DataFrame())
        assert gen._last_outcome is None

    def test_duplicate_add_data_answers_each_request_once(self):
        """Prevent duplicate submissions from shifting subsequent objective responses."""
        gen = ScipyGenerator(
            vocs=VOCS(
                variables={"x0": [-5, 5], "x1": [-5, 5]},
                objectives={"y": "MINIMIZE"},
            ),
            initial_point={"x0": 1.0, "x1": 1.0},
        )
        outcomes = []

        def minimize_sequence(objective, x0, **kwargs):
            for point in ([1.0, 1.0], [2.0, 2.0]):
                outcomes.append(objective(np.array(point)))
            return OptimizeResult(success=True, message="Finished")

        with patch("xopt.generators.sequential.scipy.minimize", minimize_sequence):
            try:
                candidate = gen.generate(1)[0]
                gen.add_data(pd.DataFrame([{**candidate, "y": 10.0}]))
                gen.add_data(pd.DataFrame([{**candidate, "y": 999.0}]))
                candidate = gen.generate(1)[0]
                gen.add_data(pd.DataFrame([{**candidate, "y": 20.0}]))
                gen._session_thread.join(timeout=2.0)
                assert outcomes == [10.0, 20.0]
                assert gen._response_queue.empty()
            finally:
                gen.reset()

    @pytest.mark.parametrize("late_action", ["objective", "result", "exception"])
    def test_timed_out_worker_cannot_poison_new_session(self, late_action):
        """Isolate stale objective calls, results, and exceptions after shutdown timeout."""
        gen = ScipyGenerator(
            vocs=VOCS(
                variables={"x0": [-5, 5], "x1": [-5, 5]},
                objectives={"y": "MINIMIZE"},
            ),
            initial_point={"x0": 1.0, "x1": 1.0},
        )
        entered = Event()
        release = Event()

        def delayed_minimize(objective, x0, **kwargs):
            entered.set()
            assert release.wait(timeout=10.0)
            if late_action == "objective":
                objective(np.array([4.0, 4.0]))
            elif late_action == "exception":
                raise ValueError("stale worker exception")
            return OptimizeResult(success=True, message="Stale result")

        old_thread = None
        new_thread = None
        queues = []
        try:
            with patch("xopt.generators.sequential.scipy.minimize", delayed_minimize):
                gen._start_session_if_needed()
                old_thread = gen._session_thread
                queues.append(gen._response_queue)
                assert entered.wait(timeout=2.0)
                with (
                    patch.object(old_thread, "join", return_value=None),
                    pytest.warns(RuntimeWarning, match="remains isolated"),
                ):
                    gen.reset()

            gen.generate(1)
            new_thread = gen._session_thread
            queues.append(gen._response_queue)
            release.set()
            old_thread.join(timeout=2.0)

            assert not old_thread.is_alive()
            assert new_thread.is_alive()
            assert gen.result is None
            assert gen._session_exception is None
            assert not gen._session_finished
            assert gen._request_queue.empty()
            assert gen._response_queue.empty()
            assert gen._point_key(np.array([4.0, 4.0])) not in gen._cache
        finally:
            release.set()
            for queue in queues:
                queue.put(gen._stop_token)
                queue.put(gen._stop_token)
            gen.reset()
            if old_thread is not None:
                old_thread.join(timeout=2.0)
            if new_thread is not None:
                new_thread.join(timeout=2.0)

    def test_requested_points_survive_readback_and_restart(self):
        """Retain requested-point answers across replay/copies until history replacement."""
        gen = ScipyGenerator(
            vocs=VOCS(
                variables={"x0": [-5, 5], "x1": [-5, 5]},
                objectives={"y": "MINIMIZE"},
            ),
            initial_point={"x0": 1.0, "x1": 1.0},
        )
        try:
            candidate = gen.generate(1)[0]
            readback = {name: value * (1.0 + 5e-7) for name, value in candidate.items()}
            gen.add_data(pd.DataFrame([{**readback, "y": 2.0}]))
            requested_key = gen._point_key(np.array(list(candidate.values())))
            assert gen._build_cache()[requested_key] == 2.0
            gen.reset()
            restored_generators = [
                gen,
                copy.deepcopy(gen),
                pickle.loads(pickle.dumps(gen)),
            ]
            for restored in restored_generators:
                try:
                    assert restored.generate(1)[0] != candidate
                finally:
                    restored.reset()
        finally:
            gen.reset()

        gen.set_data(pd.DataFrame([{**readback, "y": 3.0}]))
        assert requested_key not in gen._build_cache()
        assert gen._pending_request is None

    def test_unknown_solver_raises_validation_error(self):
        """Reject unsupported solver configuration when loading Xopt from YAML."""
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
        """Propagate unrecognized worker ValueErrors without remapping them."""
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

    def test_steps_after_scipy_termination_do_not_restart_session(self):
        """Run 200 Xopt steps with one scipy invocation, one cache build, and one warning."""
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        gen = ScipyGenerator(
            vocs=vocs,
            initial_point={"x0": 0.5, "x1": -0.5},
            options={"maxiter": 1},
        )
        xopt = Xopt(generator=gen, evaluator=Evaluator(function=sphere))
        gen = xopt.generator
        with (
            patch(
                "xopt.generators.sequential.scipy.minimize", wraps=minimize
            ) as mocked_minimize,
            patch.object(
                ScipyGenerator, "_build_cache", wraps=gen._build_cache
            ) as build_cache,
            warnings.catch_warnings(record=True) as recorded,
        ):
            warnings.simplefilter("always")
            for _ in range(200):
                xopt.step()
            mocked_minimize.assert_called_once()
            build_cache.assert_called_once()
        assert gen.is_done
        assert not gen.result.success
        assert gen.result.nit == 1
        assert len(recorded) == 1
        assert "re-evaluate" in str(recorded[0].message)
        assert len(xopt.data) == 200
        gen.reset()

    @pytest.mark.parametrize("success", [True, False])
    def test_termination_preserves_result_and_last_point(self, success):
        """Latch either termination result through copies and clear it on reset/reload."""
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        gen = ScipyGenerator(
            vocs=vocs, method="Powell", initial_point={"x0": 0.5, "x1": -0.5}
        )
        data = pd.DataFrame([{"x0": 0.1, "x1": 0.2, "y": 0.05}])
        gen._set_data(data)

        scipy_result = OptimizeResult(
            success=success, message="Optimization terminated", nit=3
        )
        assert gen.result is None
        assert not gen.is_done
        with (
            patch(
                "xopt.generators.sequential.scipy.minimize",
                return_value=scipy_result,
            ) as mocked_minimize,
            patch.object(
                ScipyGenerator, "_build_cache", wraps=gen._build_cache
            ) as build_cache,
        ):
            with pytest.warns(RuntimeWarning, match="Optimization terminated"):
                result = gen.generate(1)
            session_thread = gen._session_thread
            session_thread.join(timeout=2.0)
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                for _ in range(200):
                    assert gen.generate(1) == result
            assert not recorded
            assert gen._session_thread is session_thread
            mocked_minimize.assert_called_once()
            build_cache.assert_called_once()

        assert result is not None
        assert len(result) == 1
        assert result[0]["x0"] == pytest.approx(0.1)
        assert result[0]["x1"] == pytest.approx(0.2)
        assert gen.is_done
        assert gen.result is scipy_result
        assert gen.result.success == success
        assert gen.result.nit == 3

        for restored in (copy.deepcopy(gen), pickle.loads(pickle.dumps(gen))):
            assert restored.is_done
            assert restored.result.message == scipy_result.message
            with patch("xopt.generators.sequential.scipy.minimize") as minimize_mock:
                with warnings.catch_warnings(record=True) as recorded:
                    warnings.simplefilter("always")
                    assert restored.generate(1) == result
                assert not recorded
                minimize_mock.assert_not_called()
            restored.reset()
            assert not restored.is_done
            assert restored.result is None

        gen.reset()
        assert gen.result is None
        assert not gen.is_done
        with patch(
            "xopt.generators.sequential.scipy.minimize", return_value=scipy_result
        ) as mocked_minimize:
            with pytest.warns(RuntimeWarning):
                gen.generate(1)
            mocked_minimize.assert_called_once()
        gen.set_data(data)
        assert gen.result is None
        assert not gen.is_done

    def test_convergence_path_with_constants(self):
        """Include configured constants in the last-row termination fallback."""
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

        with patch(
            "xopt.generators.sequential.scipy.minimize",
            return_value=OptimizeResult(success=True, message="Finished"),
        ):
            with pytest.warns(RuntimeWarning):
                result = gen._generate()

        assert result is not None
        assert "c" in result[0]

    def test_convergence_path_without_data_raises_runtime_error(self):
        """Raise when scipy terminates without history for the last-row fallback."""
        vocs = VOCS(
            variables={"x0": [-5, 5], "x1": [-5, 5]},
            objectives={"y": "MINIMIZE"},
        )
        gen = ScipyGenerator(
            vocs=vocs, method="Powell", initial_point={"x0": 0.5, "x1": -0.5}
        )

        with patch(
            "xopt.generators.sequential.scipy.minimize",
            return_value=OptimizeResult(success=True, message="Finished"),
        ):
            with pytest.raises(
                RuntimeError, match="scipy minimize converged without available data"
            ):
                gen._generate()

    def test_generate_with_constants(self):
        """Include configured constants in candidates requested by the live worker."""
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
