from xopt.generators.bayesian.bayesian_exploration import BayesianExplorationGenerator
from xopt.generators.bayesian.mobo import MOBOGenerator
from xopt.generators.bayesian.upper_confidence_bound import \
    UpperConfidenceBoundGenerator
from xopt.generators.scipy.neldermead import NelderMeadGenerator

from xopt.generators.ga import CNSGAGenerator
from xopt.generators.random import RandomGenerator

# add generators here to be registered
registered_generators = [
    UpperConfidenceBoundGenerator,
    MOBOGenerator,
    BayesianExplorationGenerator,
    CNSGAGenerator,
    RandomGenerator,
    NelderMeadGenerator,
]

generators = {gen.alias: gen for gen in registered_generators}
generator_default_options = {
    gen.alias: gen.default_options() for gen in registered_generators
}
