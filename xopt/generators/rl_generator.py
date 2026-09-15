import logging
from typing import Any, ClassVar, Dict, List

import numpy as np
from pydantic import ConfigDict, model_validator

from xopt.errors import GeneratorError, VOCSError
from xopt.generator import Generator
from xopt.vocs import ContextualVariable

from gest_api.vocs import ContinuousVariable

logger = logging.getLogger(__name__)


class RLGenerator(Generator):
    """
    Generator that deploys a reinforcement-learning policy trained externally
    (outside Xopt) against a stateful evaluator such as `GymEvaluator`.

    The policy must expose a stable-baselines3 style
    ``predict(observation, deterministic) -> (action, state)`` method. Xopt does
    not train or update the policy; it is only used for inference.

    Parameters
    ----------
    policy : Any
        Trained policy object with a ``predict(observation, deterministic)`` method.
    action_space_names : List[str]
        VOCS variable names the policy's action maps to, in policy output order.
    observation_space_names : List[str]
        VOCS variable names the policy's observation is built from, in policy input order.
    initial_observation : Dict[str, float]
        Observation used to select the very first action, before any data exists.
    deterministic : bool, default=True
        Whether to sample the policy deterministically.
    """

    name: ClassVar[str] = "rl_policy"
    supports_batch_generation: bool = False
    supports_single_objective: bool = True
    supports_multi_objective: bool = False
    supports_constraints: bool = False
    supports_discrete_variables: bool = False
    supports_contextual_variables: bool = True

    policy: Any
    action_space_names: List[str]
    observation_space_names: List[str]
    initial_observation: Dict[str, float]
    deterministic: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_rl_names(self):
        for name in self.action_space_names:
            if name not in self.vocs.variables:
                raise VOCSError(f"action variable `{name}` not found in vocs.variables")
            if isinstance(self.vocs.variables[name], ContextualVariable):
                raise VOCSError(
                    f"action variable `{name}` cannot be a ContextualVariable"
                )
            if not isinstance(self.vocs.variables[name], ContinuousVariable):
                raise VOCSError(
                    f"action variable `{name}` must be a ContinuousVariable"
                )

        for name in self.observation_space_names:
            if name not in self.vocs.variables:
                raise VOCSError(
                    f"observation variable `{name}` not found in vocs.variables"
                )
            if not isinstance(self.vocs.variables[name], ContextualVariable):
                raise VOCSError(
                    f"observation variable `{name}` must be a ContextualVariable"
                )

        missing = set(self.observation_space_names) - set(self.initial_observation)
        if missing:
            raise VOCSError(f"initial_observation is missing entries for {missing}")

        return self

    def _current_observation(self) -> Dict[str, float]:
        """Latest known observation, from the last evaluated row or the initial fallback."""
        if self.data is None or len(self.data) == 0:
            return self.initial_observation

        last_row = self.data.iloc[-1]
        return {
            name: float(last_row[f"next_{name}"])
            for name in self.observation_space_names
        }

    def generate(self, n_candidates: int) -> List[Dict[str, float]]:
        if n_candidates != 1:
            raise GeneratorError(
                "RLGenerator only supports generating one candidate at a time"
            )

        observation = self._current_observation()
        obs_array = np.array(
            [observation[name] for name in self.observation_space_names]
        )

        action, _state = self.policy.predict(
            obs_array, deterministic=self.deterministic
        )

        return [
            {name: float(action[i]) for i, name in enumerate(self.action_space_names)}
        ]
