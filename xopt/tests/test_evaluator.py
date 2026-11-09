import builtins
import logging
import runpy

import pytest
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd
from pydantic import ValidationError

from xopt.vocs import random_inputs

from xopt import Evaluator, Xopt
import xopt.evaluator as evaluator_module
from xopt.errors import XoptError
from xopt.evaluator import DummyExecutor, safe_function, validate_outputs
from xopt.generators import RandomGenerator
from xopt.pydantic import NormalExecutor
from xopt.vocs import VOCS


class TestEvaluator:
    @staticmethod
    def f(x, a=True):
        if a:
            return {"f": x["x1"] ** 2 + x["x2"] ** 2}
        else:
            return {"f": False}

    @staticmethod
    def g(x, a=True):
        return False

    @staticmethod
    def identity(x, a=True):
        return {k + "_out": v for k, v in x.items()}

    def test_evaluate(self):
        # test in serial
        evaluator = Evaluator(function=self.f)
        candidates = pd.DataFrame(np.random.rand(10, 2), columns=["x1", "x2"])
        results = evaluator.evaluate_data(candidates)
        assert len(results) == 10

        # test in parallel
        evaluator = Evaluator(
            function=self.f, executor=ProcessPoolExecutor(), max_workers=2
        )
        candidates = pd.DataFrame(np.random.rand(10, 2), columns=["x1", "x2"])
        results = evaluator.evaluate_data(candidates)
        assert len(results) == 10

    def test_submit(self):
        evaluator = Evaluator(function=self.f)
        candidates = pd.DataFrame(np.random.rand(10, 2), columns=["x1", "x2"])
        futures = evaluator.submit_data(candidates)
        assert len(futures) == 10

        # try with a bad function
        evaluator = Evaluator(function=self.g)
        candidates = pd.DataFrame(np.random.rand(10, 2), columns=["x1", "x2"])
        futures = evaluator.submit_data(candidates)
        assert len(futures) == 10

    def test_type_preservation(self):
        """
        Tests for these problems:
        iterrows does not necessarily preserve types:
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iterrows.html
        itertuples does not preserve column names with understores or colons.
        """
        evaluator = Evaluator(function=self.identity)

        #  Test with ints and floats ONLY. A
        candidates = pd.DataFrame(
            {
                "ints": [1, 2, 3],
                "floats": [2.0, 3.0, 4.0],
                "_leading": [2.0, 3.0, 4.0],
                "has:colon": [1, 2.0, 3],
                ":colon:leading": [2.0, 3.0, 4.0],
                "strings": ["a", "b", "c"],
                "booleans": [True, False, True],
                "lists": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "dicts": [{"a": 1, "b": 2}, {"c": 3, "d": 4}, {"e": 5, "f": 6}],
                "dicts_compund": [
                    {"a": 1, "b": True},
                    {"c": 3, "d": "4"},
                    {"e": 5, "f": 6},
                ],
                "tuples": [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
            },
            index=[7, 8, 9],
        )
        futures = evaluator.submit_data(candidates)
        data = [fut.result() for fut in futures]
        df2 = pd.DataFrame(data, index=candidates.index)

        # Strip additional Xopt columns
        for key in df2.columns:
            if key.startswith("xopt_"):
                df2.pop(key)

        df2.columns = df2.columns.str.replace("_out", "")
        assert df2.equals(candidates), "DataFrame types were not preserved"

    def test_init(self):
        test_dict = {
            "function": self.f,
            "max_workers": 1,
        }
        evaluator = Evaluator(**test_dict)
        candidates = pd.DataFrame(np.random.rand(10, 2), columns=["x1", "x2"])
        futures = evaluator.submit_data(candidates)
        assert len(futures) == 10

        evaluator = Evaluator(**test_dict)
        candidates = pd.DataFrame(np.random.rand(10, 2), columns=["x1", "x2"])
        futures = evaluator.submit_data(candidates)
        assert len(futures) == 10

        test_dict["function_kwargs"] = {"a": False}
        evaluator = Evaluator(**test_dict)
        candidates = pd.DataFrame(np.random.rand(10, 2), columns=["x1", "x2"])
        futures = evaluator.submit_data(candidates)
        assert len(futures) == 10
        assert futures[0].result()["f"] is False

    def test_serialize(self):
        test_dict = {
            "function": self.f,
            "max_workers": 1,
        }

        for Executor in [ThreadPoolExecutor, ProcessPoolExecutor]:
            test_dict["executor"] = Executor()
            ev = Evaluator(**test_dict)
            ev.model_dump_json()

    def test_multithreading(self):
        MAX_WORKERS = 10

        vocs = VOCS(variables={"x": [0, 1], "y": [0, 1]}, objectives={"f1": "MINIMIZE"})
        in10 = random_inputs(vocs, 10)

        # Create Executor insance
        executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
        ev = Evaluator(function=self.f, executor=executor, max_workers=MAX_WORKERS)
        ev.evaluate_data(in10)

    def test_evaluate_return_types(self):
        def a(input):
            return {"f": 1.0}

        def b(input):
            return {"f": [1.0]}

        def c(input):
            return {"f": np.array(1.0)}

        def d(input):
            return {"f": [1.0, 1.0]}

        def e(input):
            return {"f": np.ones(3)}

        vocs = VOCS(variables={"x": [0, 1]}, objectives={"f": "MINIMIZE"})

        for func in [a, b, c, d, e]:
            evaluator = Evaluator(function=func)
            generator = RandomGenerator(vocs=vocs)

            X = Xopt(evaluator=evaluator, generator=generator)

            X.random_evaluate(5)

    def test_submit_non_dict(self):
        evaluator = Evaluator(function=self.f)
        with pytest.raises(ValueError):
            evaluator.submit([1, 2, 3])
        with pytest.raises(ValueError):
            evaluator.submit("not a dict")
        with pytest.raises(ValueError):
            evaluator.submit(42)

    def test_submit_data_vectorized(self):
        evaluator = Evaluator(function=self.f, vectorized=True)
        candidates = pd.DataFrame(np.random.rand(5, 2), columns=["x1", "x2"])
        futures = evaluator.submit_data(candidates)
        assert len(futures) == 1
        result = futures[0].result()
        assert "f" in result

    def test_evaluate_data_vectorized(self):
        evaluator = Evaluator(function=self.f, vectorized=True)
        candidates = pd.DataFrame(np.random.rand(5, 2), columns=["x1", "x2"])

        result = evaluator.evaluate_data(candidates)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 5
        assert "f" in result.columns
        assert all(result["f"] == candidates["x1"] ** 2 + candidates["x2"] ** 2)

    @pytest.mark.parametrize(
        "inputs",
        [
            {"x1": 2.0, "x2": 3.0},
            {"x1": [2.0], "x2": [3.0]},
            [{"x1": 2.0, "x2": 3.0}],
        ],
    )
    def test_evaluate_data_input_formats(self, inputs):
        result = Evaluator(function=self.f).evaluate_data(inputs)
        assert result["f"].tolist() == [13.0]
        assert not result["xopt_error"].any()

    @pytest.mark.parametrize("conflicting", [False, True])
    @pytest.mark.parametrize("vectorized", [False, True])
    def test_evaluate_data_duplicate_columns(self, conflicting, vectorized):
        def function(inputs):
            return {"x1": inputs["x1"] + int(conflicting), "f": inputs["x1"] ** 2}

        evaluator = Evaluator(function=function, vectorized=vectorized)
        inputs = pd.DataFrame({"x1": [2.0, 3.0]}, index=[4, 7])
        if conflicting:
            with pytest.raises(ValueError, match="output column 'x1' conflicts"):
                evaluator.evaluate_data(inputs)
        else:
            result = evaluator.evaluate_data(inputs)
            assert result.columns.is_unique
            pd.testing.assert_series_equal(result["x1"], inputs["x1"])
            assert result["f"].tolist() == [4.0, 9.0]

    def test_evaluate_overrides_function_kwargs(self):
        evaluator = Evaluator(function=self.f, function_kwargs={"a": False})
        assert evaluator.evaluate({"x1": 2.0, "x2": 3.0})["f"] is False
        assert evaluator.evaluate({"x1": 2.0, "x2": 3.0}, a=True)["f"] == 13.0
        assert evaluator.function_kwargs == {"a": False}

    def test_reuses_wrapped_executor(self):
        with ThreadPoolExecutor(max_workers=1) as executor:
            wrapped = NormalExecutor[ThreadPoolExecutor](executor=executor)
            evaluator = Evaluator(function=self.f, executor=wrapped)
            assert evaluator.executor is wrapped
            assert evaluator.submit({"x1": 2.0, "x2": 3.0}).result()["f"] == 13.0

    def test_default_parallel_executor(self):
        evaluator = Evaluator(function=self.f, max_workers=2)
        try:
            assert isinstance(evaluator.executor.executor, ProcessPoolExecutor)
            result = evaluator.evaluate_data(pd.DataFrame({"x1": [2.0], "x2": [3.0]}))
            assert result["f"].tolist() == [13.0]
        finally:
            evaluator.executor.shutdown()


class TestSafeFunction:
    def test_success(self):
        result = safe_function(lambda x, offset: {"f": x + offset}, 2, offset=3)
        assert result["f"] == 5
        assert result["xopt_error"] is False
        assert result["xopt_runtime"] >= 0
        assert "xopt_error_str" not in result
        validate_outputs(pd.DataFrame([result]))

    @pytest.mark.parametrize("value", [None, False, 42, [1, 2]])
    def test_non_dict_result(self, value):
        result = safe_function(lambda: value)
        assert result["xopt_non_dict_result"] == value
        assert result["xopt_error"] is True
        assert result["xopt_error_str"] == "Non-dict result"
        with pytest.raises(XoptError, match="returned a non-dict result"):
            validate_outputs(pd.DataFrame([result]))

    def test_exception(self):
        def fail():
            raise RuntimeError("evaluation failed")

        result = safe_function(fail)
        assert result["xopt_error"] is True
        assert "RuntimeError: evaluation failed" in result["xopt_error_str"]
        assert "xopt_non_dict_result" not in result
        with pytest.raises(XoptError) as exc_info:
            validate_outputs(pd.DataFrame([result, result]))
        message = str(exc_info.value)
        assert "Evaluation index 0:" in message
        assert "Evaluation index 1:" in message
        assert message.count("RuntimeError: evaluation failed") == 2


class TestDummyExecutor:
    def test_submit_and_map(self):
        with DummyExecutor() as executor:
            future = executor.submit(pow, 2, exp=3)
            assert future.done()
            assert future.result() == 8
            assert list(executor.map(pow, [2, 3], [3, 2])) == [8, 9]

    def test_submit_exception(self):
        def fail():
            raise RuntimeError("task failed")

        with DummyExecutor() as executor:
            future = executor.submit(fail)
            assert future.done()
            with pytest.raises(RuntimeError, match="task failed"):
                future.result()

    def test_submit_after_shutdown(self):
        executor = DummyExecutor()
        executor.shutdown()
        executor.shutdown()
        with pytest.raises(
            RuntimeError, match="cannot schedule new futures after shutdown"
        ):
            executor.submit(pow, 2, 3)


class TestGymEvaluator:
    @pytest.fixture
    def gym_evaluator(self):
        gym = pytest.importorskip("gymnasium")
        from xopt.evaluator import GymEvaluator

        env = gym.make("Pendulum-v1")
        try:
            yield GymEvaluator(
                env=env,
                action_space_names=["torque"],
                observation_space_names=["cos_theta", "sin_theta", "theta_dot"],
            )
        finally:
            env.close()

    def test_import_without_gymnasium(self, monkeypatch, caplog):
        original_import = builtins.__import__

        def import_without_gymnasium(name, *args, **kwargs):
            if name == "gymnasium":
                raise ModuleNotFoundError("No module named 'gymnasium'")
            return original_import(name, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", import_without_gymnasium)
            with caplog.at_level(logging.DEBUG):
                namespace = runpy.run_path(evaluator_module.__file__)

        assert namespace["_HAS_GYMNASIUM"] is False
        assert "GymEvaluator" not in namespace
        assert "gymnasium not installed, GymEvaluator is not available" in caplog.text

    @pytest.mark.parametrize("max_workers", [-1, 0, 2])
    def test_max_workers_validation(self, max_workers):
        gym = pytest.importorskip("gymnasium")
        from xopt.evaluator import GymEvaluator

        env = gym.make("Pendulum-v1")
        try:
            kwargs = {
                "env": env,
                "action_space_names": ["torque"],
                "observation_space_names": ["cos_theta", "sin_theta", "theta_dot"],
            }
            with pytest.raises(
                ValidationError, match=r"max_workers\s+Input should be 1"
            ):
                GymEvaluator(**kwargs, max_workers=max_workers)

            evaluator = GymEvaluator(**kwargs, max_workers=1)
            assert evaluator.max_workers == 1
            executor = evaluator.executor
            with pytest.raises(
                ValidationError, match=r"max_workers\s+Input should be 1"
            ):
                evaluator.max_workers = max_workers
            assert evaluator.max_workers == 1
            assert evaluator.executor is executor

            evaluator.max_workers = 1
            evaluator.reset()
            assert evaluator.executor is executor
            assert "reward" in evaluator.evaluate({"torque": 0.0})
        finally:
            env.close()

    @pytest.mark.parametrize(
        "terminated,truncated", [(False, False), (True, False), (False, True)]
    )
    def test_step_output_and_history(
        self, gym_evaluator, monkeypatch, terminated, truncated
    ):
        evaluator = gym_evaluator
        before = evaluator.current_observation
        observation = np.array([0.6, 0.8, 0.2])
        info = {"step": 1}
        actions = []

        def step(action):
            actions.append(action.copy())
            return observation, -2.5, terminated, truncated, info

        monkeypatch.setattr(evaluator.env, "step", step)
        result = evaluator.evaluate({"torque": 0.5})
        assert result["xopt_error"] is False
        assert {name: result[name] for name in before} == before
        assert evaluator.current_observation == {
            "cos_theta": 0.6,
            "sin_theta": 0.8,
            "theta_dot": 0.2,
        }
        for name, value in evaluator.current_observation.items():
            assert result[f"next_{name}"] == value
        assert result["reward"] == -2.5
        assert result["terminated"] is terminated
        assert result["truncated"] is truncated
        assert result["info"] == info
        np.testing.assert_array_equal(actions, [[0.5]])
        np.testing.assert_array_equal(evaluator.action_history, [[0.5]])

        second = evaluator.evaluate({"torque": -0.5})
        for name, value in evaluator.current_observation.items():
            assert second[name] == value
        np.testing.assert_array_equal(evaluator.action_history, [[0.5], [-0.5]])

    def test_reset_clears_history(self, gym_evaluator, monkeypatch):
        evaluator = gym_evaluator
        evaluator.evaluate({"torque": 0.5})
        executor = evaluator.executor
        monkeypatch.setattr(
            evaluator.env, "reset", lambda: (np.array([1.0, 0.0, 0.0]), {})
        )
        evaluator.reset()
        assert evaluator.action_history == []
        assert evaluator.current_observation == {
            "cos_theta": 1.0,
            "sin_theta": 0.0,
            "theta_dot": 0.0,
        }
        assert evaluator.executor is executor

    def test_environment_action_mode(self, monkeypatch):
        gym = pytest.importorskip("gymnasium")
        from xopt.evaluator import GymEvaluator

        with gym.make("Pendulum-v1") as env:
            monkeypatch.setattr(env, "action_mode", "absolute", raising=False)
            evaluator = GymEvaluator(
                env=env,
                action_space_names=["torque"],
                observation_space_names=["cos_theta", "sin_theta", "theta_dot"],
            )
            assert evaluator.action_mode == "absolute"

    def test_default_workers_and_independent_instances(self, gym_evaluator):
        gym = pytest.importorskip("gymnasium")
        from xopt.evaluator import GymEvaluator

        with gym.make("Pendulum-v1") as env:
            other = GymEvaluator(
                env=env,
                action_space_names=["torque"],
                observation_space_names=["cos_theta", "sin_theta", "theta_dot"],
            )
            assert gym_evaluator.max_workers == other.max_workers == 1
            assert gym_evaluator.action_mode == other.action_mode == "delta"
            gym_evaluator.evaluate({"torque": 0.5})
            assert len(gym_evaluator.action_history) == 1
            assert other.action_history == []
            assert other.executor is not gym_evaluator.executor

    def test_current_observation_and_step(self):
        gym = pytest.importorskip("gymnasium")
        from xopt.evaluator import GymEvaluator

        env = gym.make("Pendulum-v1")
        obs_names = ["cos_theta", "sin_theta", "theta_dot"]
        evaluator = GymEvaluator(
            env=env,
            action_space_names=["torque"],
            observation_space_names=obs_names,
        )

        # current_observation should match the reset observation
        env.reset(seed=0)
        evaluator.reset()
        assert evaluator.current_observation == {
            name: pytest.approx(float(evaluator._current_observation[i]))
            for i, name in enumerate(obs_names)
        }

        result = evaluator.evaluate({"torque": 0.0})
        assert set(obs_names).issubset(result)
        assert {f"next_{n}" for n in obs_names}.issubset(result)
        assert "reward" in result

        # current_observation should now reflect the post-step state
        assert evaluator.current_observation == {
            name: pytest.approx(result[f"next_{name}"]) for name in obs_names
        }
