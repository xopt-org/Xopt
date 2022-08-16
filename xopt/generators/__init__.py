from xopt.generators.bayesian.bayesian_exploration import BayesianExplorationGenerator
from xopt.generators.bayesian.mobo import MOBOGenerator
from xopt.generators.bayesian.upper_confidence_bound import \
    UpperConfidenceBoundGenerator, TDUpperConfidenceBoundGenerator
from xopt.generators.bayesian.expected_improvement import ExpectedImprovementGenerator
from xopt.generators.scipy.neldermead import NelderMeadGenerator
from xopt.generators.bayesian.mggpo import MGGPOGenerator

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
    TDUpperConfidenceBoundGenerator,
    ExpectedImprovementGenerator,
    MGGPOGenerator
]

generators = {gen.alias: gen for gen in registered_generators}
generator_default_options = {
    gen.alias: gen.default_options() for gen in registered_generators
}
