import numpy as np
import pytest

from xopt import Xopt
from xopt.errors import GeneratorError, VOCSError
from xopt.generators.rl_generator import RLGenerator
from xopt.vocs import ContextualVariable

from gest_api.vocs import VOCS


class StubPolicy:
    """Fixed-action stand-in for a trained stable-baselines3 policy."""

    def predict(self, observation, deterministic=True):
        return np.array([0.0]), None


def _build_vocs():
    return VOCS(
        variables={
            "torque": [-2.0, 2.0],
            "cos_theta": ContextualVariable(),
            "sin_theta": ContextualVariable(),
            "theta_dot": ContextualVariable(),
        },
        objectives={"reward": "MAXIMIZE"},
    )


class TestRLGenerator:
    def test_generate_uses_initial_observation(self):
        vocs = _build_vocs()
        obs_names = ["cos_theta", "sin_theta", "theta_dot"]
        generator = RLGenerator(
            vocs=vocs,
            policy=StubPolicy(),
            action_space_names=["torque"],
            observation_space_names=obs_names,
            initial_observation={"cos_theta": 1.0, "sin_theta": 0.0, "theta_dot": 0.0},
        )

        candidates = generator.generate(1)
        assert candidates == [{"torque": 0.0}]

    def test_generate_rejects_batch(self):
        vocs = _build_vocs()
        generator = RLGenerator(
            vocs=vocs,
            policy=StubPolicy(),
            action_space_names=["torque"],
            observation_space_names=["cos_theta", "sin_theta", "theta_dot"],
            initial_observation={"cos_theta": 1.0, "sin_theta": 0.0, "theta_dot": 0.0},
        )
        with pytest.raises(GeneratorError):
            generator.generate(2)

    def test_action_name_must_be_continuous(self):
        vocs = _build_vocs()
        with pytest.raises(VOCSError):
            RLGenerator(
                vocs=vocs,
                policy=StubPolicy(),
                action_space_names=["cos_theta"],
                observation_space_names=["sin_theta", "theta_dot"],
                initial_observation={"sin_theta": 0.0, "theta_dot": 0.0},
            )

    def test_observation_name_must_be_contextual_variable(self):
        vocs = _build_vocs()
        with pytest.raises(VOCSError):
            RLGenerator(
                vocs=vocs,
                policy=StubPolicy(),
                action_space_names=["torque"],
                observation_space_names=["torque"],
                initial_observation={"torque": 0.0},
            )

    def test_run_with_gym_evaluator(self):
        gym = pytest.importorskip("gymnasium")
        from xopt.evaluator import GymEvaluator

        vocs = _build_vocs()
        obs_names = ["cos_theta", "sin_theta", "theta_dot"]
        env = gym.make("Pendulum-v1")
        evaluator = GymEvaluator(
            env=env,
            action_space_names=["torque"],
            observation_space_names=obs_names,
            max_workers=1,
        )
        generator = RLGenerator(
            vocs=vocs,
            policy=StubPolicy(),
            action_space_names=["torque"],
            observation_space_names=obs_names,
            initial_observation=evaluator.current_observation,
        )

        X = Xopt(generator=generator, evaluator=evaluator)
        for _ in range(3):
            X.step()

        assert len(X.data) == 3
        for name in obs_names:
            assert f"next_{name}" in X.data.columns
        assert all(X.data["torque"] == 0.0)
