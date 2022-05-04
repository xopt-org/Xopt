from .upper_confidence_bound import UpperConfidenceBoundGenerator, UpperConfidenceBoundOptions
from .bayesian_exploration import BayesianExplorationGenerator
from .bayesian_generator import BayesianGenerator, BayesianOptions
from .mobo import MOBOGenerator, MOBOOptions

registry = {
    "UpperConfidenceBound": (UpperConfidenceBoundGenerator,
                             UpperConfidenceBoundOptions),
    "MOBO": (MOBOGenerator, MOBOOptions),
    "BayesianExploration": (BayesianExplorationGenerator, BayesianOptions)
}

