from xopt.generators.bayesian.models.multi_fidelity import MultiFidelityModelConstructor
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.bayesian.models.time_dependent import TimeDependentModelConstructor


def get_model_constructor(model_options):
    if model_options.custom_constructor:
        constructor = model_options.custom_constructor
    else:
        name = model_options.name
        if name == "standard":
            constructor = StandardModelConstructor
        elif name == "time_dependent_standard":
            constructor = TimeDependentModelConstructor
        elif name == "multi_fidelity":
            constructor = MultiFidelityModelConstructor
        else:
            raise ValueError(f"{name} is not a vaild model name")

    return constructor
