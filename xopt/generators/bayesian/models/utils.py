from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.bayesian.models.time_dependent import TimeDependentModelConstructor
from xopt.generators.bayesian.models.trainable_mean import TrainableMeanModelConstructor


def get_model_constructor(model_options):
    if model_options.custom_constructor:
        constructor = model_options.custom_constructor
    else:
        name = model_options.name
        if name == "standard":
            constructor = StandardModelConstructor
        elif name == "time_dependent_standard":
            constructor = TimeDependentModelConstructor
        elif name == "trainable_mean_standard":
            constructor = TrainableMeanModelConstructor
        else:
            raise ValueError(f"{name} is not a vaild model name")

    return constructor
