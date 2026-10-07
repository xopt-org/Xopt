"""
Reinforcement-learning example: swing up an inverted pendulum with Xopt.

This script has two independent parts:

1. ``train_policy`` trains a SAC policy on gymnasium's ``Pendulum-v1`` with
   stable-baselines3, entirely outside of Xopt (the policy is just saved to disk).
2. ``run_with_xopt`` loads that frozen policy and deploys it through
   ``RLGenerator`` + ``GymEvaluator`` to run one Xopt-driven rollout.

Run with: ``python docs/examples/rl/inverted_pendulum.py``
Requires the optional ``rl`` extra: ``pip install xopt[rl]``.
"""

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from gest_api.vocs import VOCS
from xopt import Xopt
from xopt.evaluator import GymEvaluator
from xopt.generators.rl_generator import RLGenerator
from xopt.vocs import ContextualVariable

MODEL_PATH = Path(__file__).parent / "sac_pendulum.zip"
ACTION_NAMES = ["torque"]
OBSERVATION_NAMES = ["cos_theta", "sin_theta", "theta_dot"]


def train_policy() -> SAC:
    """Train (or load a previously cached) SAC policy on Pendulum-v1."""
    if MODEL_PATH.exists():
        return SAC.load(MODEL_PATH)

    env = gym.make("Pendulum-v1")
    model = SAC("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=20_000)
    model.save(MODEL_PATH)
    env.close()
    return model


def run_with_xopt(policy: SAC, n_steps: int = 200):
    """Deploy the frozen policy inside Xopt via RLGenerator + GymEvaluator."""
    vocs = VOCS(
        variables={
            "torque": [-2.0, 2.0],
            "cos_theta": ContextualVariable(),
            "sin_theta": ContextualVariable(),
            "theta_dot": ContextualVariable(),
        },
        objectives={"reward": "MAXIMIZE"},
    )

    env = gym.make("Pendulum-v1")
    evaluator = GymEvaluator(
        env=env,
        action_space_names=ACTION_NAMES,
        observation_space_names=OBSERVATION_NAMES,
        max_workers=1,
    )
    generator = RLGenerator(
        vocs=vocs,
        policy=policy,
        action_space_names=ACTION_NAMES,
        observation_space_names=OBSERVATION_NAMES,
        initial_observation=evaluator.current_observation,
    )

    X = Xopt(generator=generator, evaluator=evaluator)
    for _ in range(n_steps):
        X.step()

    env.close()
    return X.data


if __name__ == "__main__":
    trained_policy = train_policy()
    data = run_with_xopt(trained_policy)

    print(f"ran {len(data)} steps, cumulative reward = {data['reward'].sum():.2f}")

    plt.plot(data["reward"].cumsum())
    plt.xlabel("step")
    plt.ylabel("cumulative reward")
    plt.title("Inverted pendulum: RLGenerator rollout with a frozen SAC policy")
    plot_path = Path(__file__).parent / "inverted_pendulum_reward.png"
    plt.savefig(plot_path)
    print(f"saved plot to {plot_path}")
