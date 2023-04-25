from pydantic import Field

from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.bayesian.models.prior_mean import CustomMean
from xopt.generators.bayesian.options import ModelOptions
from xopt.vocs import VOCS


class TrainableMeanModelOptions(ModelOptions):
    name: str = "trainable_mean_standard"
    added_time: float = Field(
        1.0,
        description="additional time added to target time for optimization, "
                    "make sure its larger than computation time for the "
                    "GP model",
    )


class TrainableMeanModelConstructor(StandardModelConstructor):
    def __init__(self, vocs: VOCS, options: TrainableMeanModelOptions):
        if not type(options) is TrainableMeanModelOptions:
            raise ValueError(
                "options must be a TrainableMeanModelOptions object"
            )
        super().__init__(vocs, options)

    def build_mean_module(self, name, outcome_transform):
        mean_module = self._get_module(self.options.mean_modules, name)
        if mean_module is not None:
            mean_module = CustomMean(mean_module, self.input_transform,
                                     outcome_transform, training_intended=True)
        return mean_module
