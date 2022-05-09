from xopt.generators.bayesian import (
    BayesianExplorationGenerator,
    MOBOGenerator,
    UpperConfidenceBoundGenerator,
)

from xopt.generators.ga import CNSGAGenerator
from xopt.generators.random import RandomGenerator

# add generators here to be registered
registered_generators = [
    UpperConfidenceBoundGenerator,
    MOBOGenerator,
    BayesianExplorationGenerator,
    # CNSGAGenerator,
    RandomGenerator,
]

generators = {gen.alias: gen for gen in registered_generators}
generator_default_options = {
    gen.alias: gen.default_options() for gen in registered_generators
}
