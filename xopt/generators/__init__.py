from xopt.errors import XoptError
from xopt.generators.bayesian.bayesian_exploration import BayesianExplorationGenerator
from xopt.generators.bayesian.mobo import MOBOGenerator
from xopt.generators.bayesian.upper_confidence_bound import \
    UpperConfidenceBoundGenerator, TDUpperConfidenceBoundGenerator
from xopt.generators.bayesian.expected_improvement import ExpectedImprovementGenerator
from xopt.generators.scipy.neldermead import NelderMeadGenerator
from xopt.generators.es.extremumseeking import ExtremumSeekingGenerator
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
    ExtremumSeekingGenerator,
    MGGPOGenerator,
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
