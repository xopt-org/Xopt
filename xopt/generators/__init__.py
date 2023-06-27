import warnings

from xopt.errors import XoptError
from xopt.generators.es.extremumseeking import ExtremumSeekingGenerator
from xopt.generators.random import RandomGenerator
from xopt.generators.rcds.rcds import RCDSGenerator

# add generators here to be registered
registered_generators = [
    RandomGenerator,
    ExtremumSeekingGenerator,
    RCDSGenerator,
]

# generators needing deap
try:
    from xopt.generators.ga import CNSGAGenerator

    registered_generators += [CNSGAGenerator]

except ModuleNotFoundError:
    warnings.warn("WARNING: `deap` not found, CNSGAGenerator is not available")

# generators needing botorch
try:
    from xopt.generators.bayesian.bayesian_exploration import (
        BayesianExplorationGenerator,
    )
    from xopt.generators.bayesian.expected_improvement import (
        ExpectedImprovementGenerator,
    )
    from xopt.generators.bayesian.mobo import MOBOGenerator
    from xopt.generators.bayesian.multi_fidelity import MultiFidelityGenerator
    from xopt.generators.bayesian.upper_confidence_bound import (
        TDUpperConfidenceBoundGenerator,
        UpperConfidenceBoundGenerator,
    )

    registered_generators += [
        UpperConfidenceBoundGenerator,
        MOBOGenerator,
        BayesianExplorationGenerator,
        TDUpperConfidenceBoundGenerator,
        ExpectedImprovementGenerator,
        MultiFidelityGenerator,
    ]

except ModuleNotFoundError:
    warnings.warn("WARNING: `botorch` not found, Bayesian generators are not available")

# generators requiring deap AND botorch
try:
    from xopt.generators.bayesian.mggpo import MGGPOGenerator

    registered_generators += [MGGPOGenerator]
except ModuleNotFoundError:
    warnings.warn(
        "WARNING: `deap` and `botorch` not found, MGGPOGenerator is not " "available"
    )

try:
    from xopt.generators.scipy.neldermead import NelderMeadGenerator

    registered_generators += [NelderMeadGenerator]

except ModuleNotFoundError:
    warnings.warn("WARNING: `scipy` not found, NelderMeadGenerator is not available")


generators = {gen.name: gen for gen in registered_generators}


def get_generator(name: str):
    try:
        return generators[name]
    except KeyError:
        raise XoptError(
            f"No generator named {name}, available generators are {list(generators.keys())}"
        )


#
# def get_generator_help(name):
#     generator = get_generator(name)
#     help_string = f"Generator name: {generator.name}\n"
#     help_string = recursive_description(generator, help_string)
#
#     return help_string
#
#
# def recursive_description(cls, help_string, in_key="", indent_level=0):
#     help_string += f"{in_key} : {cls}\n"
#     for key, val in cls.__fields__.items():
#         try:
#             if issubclass(val.type_, BaseModel):
#                 help_string = recursive_description(val.type_, help_string, key,
#                                                     indent_level+1)
#             else:
#                 help_string += "\t"*indent_level + f"{key} ({val.type_}):" \
#                                                    f" {val.field_info.description}\n"
#         except TypeError:
#             help_string += "\t"*indent_level + f"{key} ({val.type_}):" \
#                                                f" {val.field_info.description}\n"
#
#     return help_string
