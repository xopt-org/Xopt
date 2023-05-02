from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.bayesian.models.prior_mean import CustomMean
from xopt.generators.bayesian.options import ModelOptions
from xopt.vocs import VOCS


class TrainableMeanModelConstructor(StandardModelConstructor):
    def __init__(self, vocs: VOCS, options: ModelOptions):
        if not type(options) is ModelOptions:
            raise ValueError(
                "options must be a ModelOptions object"
            )
        super().__init__(vocs, options)

    def build_mean_module(self, name, outcome_transform):
        mean_module = self._get_module(self.options.mean_modules, name)
        if mean_module is not None:
            mean_module = CustomMean(mean_module, self.input_transform,
                                     outcome_transform, fixed_model=False)
        return mean_module
