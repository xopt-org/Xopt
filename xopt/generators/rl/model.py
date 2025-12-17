import pandas as pd
from xopt.generator import Generator
from stable_baselines3.common.base_class import BaseAlgorithm
from typing import List, Dict, Any, Union


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
    """
    RL Model generator. Utilizes an RL model to generate the set of points for evaluation.
    This is configured to handle multi-variable optimization.
    """

    name = "rl_model_generator"
    supports_batch_generation: bool = True
    supports_multi_objective: bool = True
    supports_single_objective: bool = True
    rl_model: ModelWrapper = None

    def set_model(self, model: BaseAlgorithm):
        self.rl_model = ModelWrapper(model)

    @property
    def model_input_names(self) -> List[str]:
        """variable names corresponding to trained model"""
        return self.vocs.variable_names

    def generate(self, n_candidates: int) -> List[Dict[str, Union[float, Any]]]:
        if self.data is None or len(self.data) == 0:
            raise RuntimeError(
                "no data contained in generator, call `set_data` "
                "method to set data, see also `Xopt.random_evaluate()`"
            )

        model = self.rl_model.model

        input_data = self.data[self.model_input_names]

        current_states_array = input_data.values

        candidates = []
        for state in current_states_array:
            action, _ = model.predict(state, deterministic=True)

            if action.ndim > 1:
                action = action.flatten()

            new_candidate_state = state + action

            candidates.append(new_candidate_state)

        results = self.vocs.convert_numpy_to_inputs(candidates, include_constants=False)

        return results

    def set_data(self, data: pd.DataFrame):
        self.data = data
