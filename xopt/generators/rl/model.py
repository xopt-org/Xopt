import pandas as pd
from xopt.generator import Generator
from stable_baselines3.common.base_class import BaseAlgorithm
from typing import List, Dict, Any, Union
import numpy as np


class ModelWrapper:
    def __init__(self, model: BaseAlgorithm):
        self.model = model

    def __deepcopy__(self, _memo):
        """
        Tells copy.deepcopy to return the existing instance (self)
        instead of creating a new copy. This is because deepcopy doesn't work with BaseAlgorithm
        """
        return self


class RLModelGenerator(Generator):
    name = "rl_model_generator"
    supports_batch_generation: bool = True
    supports_multi_objective: bool = True
    supports_single_objective: bool = True
    rl_model: ModelWrapper = None

    def set_model(self, model: BaseAlgorithm):
        # TODO should verify the model adheres to the inputs/output expectations in vocs
        self.rl_model = ModelWrapper(model)

    @property
    def model_input_names(self) -> List[str]:
        return self.vocs.variable_names

    def generate(self, n_candidates: int) -> List[Dict[str, Union[float, Any]]]:
        if self.data is None or len(self.data) == 0:
            raise RuntimeError(
                "no data contained in generator, call `set_data` "
                "method to set data, see also `Xopt.random_evaluate()`"
            )

        bounds = np.array(
            [self.vocs.variables[name] for name in self.model_input_names]
        )
        mins = bounds[:, 0]
        maxs = bounds[:, 1]

        model = self.rl_model.model
        input_data = self.data[self.model_input_names]

        current_state = input_data.iloc[-1].values

        if current_state.ndim == 0:
            current_state = np.array([current_state])

        candidates = []

        for _ in range(n_candidates):
            action, _states = model.predict(current_state, deterministic=False)

            if action.ndim > 1:
                action = action.flatten()

            new_candidate_state = current_state + action
            new_candidate_state = np.clip(new_candidate_state, mins, maxs)

            candidates.append(new_candidate_state)

            current_state = new_candidate_state

        results = self.vocs.convert_numpy_to_inputs(candidates, include_constants=False)
        return results

    def set_data(self, data: pd.DataFrame):
        self.data = data
