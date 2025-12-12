from xopt.generator import Generator
from stable_baselines3.common.base_class import BaseAlgorithm


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
    """RL Model generator. Utilizes an RL model to generate the set of points for evaluation


    Turn this RL model params into VOCS
        * Alternative: separately define. Ensure consistency?
    """

    name = "rl_model_generator"
    supports_batch_generation: bool = True
    supports_multi_objective: bool = True
    supports_single_objective: bool = True
    rl_model: ModelWrapper = None

    def set_model(self, model):
        self.rl_model = ModelWrapper(model)

    @property
    def model_input_names(self):
        """variable names corresponding to trained model"""
        variable_names = self.vocs.variable_names
        return variable_names

    def generate(self, n_candidates: int):
        if self.data is None:
            raise RuntimeError(
                "no data contained in generator, call `set_data` "
                "method to set data, see also `Xopt.random_evaluate()`"
            )

        model = self.rl_model.model

        input_data = self.data[self.model_input_names]

        candidates = [
            input_data.values[i] + model.predict(input_data.values[i])[0]
            for i in range(n_candidates)
        ]  # TODO: Review this

        results = self.vocs.convert_numpy_to_inputs(candidates, include_constants=False)

        return results

    def set_data(self, data):
        self.data = data
