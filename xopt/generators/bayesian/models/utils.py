from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.bayesian.models.time_dependent import TimeDependentModelConstructor


def split_data(data, vocs):
    variable_data = vocs.variable_data(data, "")
    objective_data = vocs.objective_data(data, "")
    constraint_data = vocs.constraint_data(data, "")
    return variable_data, objective_data, constraint_data


def get_model_constructor(model_options):
    if model_options.custom_constructor:
        constructor = model_options.custom_constructor
    else:
        name = model_options.name
        if name == "standard":
            constructor = StandardModelConstructor
        elif name == "time_dependent_standard":
            constructor = TimeDependentModelConstructor
        else:
            raise ValueError(f"{name} is not a vaild model name")

    return constructor
