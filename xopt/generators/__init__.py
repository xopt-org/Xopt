from xopt.errors import XoptError

registered_generators = []

# generators needing numpy
from xopt.generators.random import RandomGenerator

# generators needing deap
try:
    from xopt.generators.ga import CNSGAGenerator

    registered_generators += [CNSGAGenerator]

except ModuleNotFoundError:
    print("WARNING: `deap` not found, CNSGAGenerator is not available")

# generators needing botorch
try:
    from xopt.generators.bayesian.bayesian_exploration import (
        BayesianExplorationGenerator,
    )
    from xopt.generators.bayesian.expected_improvement import (
        ExpectedImprovementGenerator,
    )
    from xopt.generators.bayesian.mobo import MOBOGenerator
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
    ]

except ModuleNotFoundError:
    print("WARNING: `botorch` not found, Bayesian generators are not available")

# generators requiring deap AND botorch
try:
    from xopt.generators.bayesian.mggpo import MGGPOGenerator

    registered_generators += [MGGPOGenerator]
except ModuleNotFoundError:
    print("WARNING: `deap` and `botorch` not found, MGGPOGenerator is not available")

from xopt.generators.es.extremumseeking import ExtremumSeekingGenerator
from xopt.generators.scipy.neldermead import NelderMeadGenerator

# add generators here to be registered
registered_generators += [
    RandomGenerator,
    NelderMeadGenerator,
    ExtremumSeekingGenerator,
]

generators = {gen.alias: gen for gen in registered_generators}
generator_default_options = {
    gen.alias: gen.default_options() for gen in registered_generators
}


def get_generator_and_defaults(name: str):
    try:
        return generators[name], generator_default_options[name]
    except KeyError:
        raise XoptError(
            f"No generator named {name}, available generators are {list(generators.keys())}"
        )
