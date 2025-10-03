from xopt.generators.bayesian.bayesian_exploration import BayesianExplorationGenerator
from xopt.generators.bayesian.expected_improvement import (
    ExpectedImprovementGenerator,
    TDExpectedImprovementGenerator,
)
from xopt.generators.bayesian.mobo import MOBOGenerator
from xopt.generators.bayesian.multi_fidelity import MultiFidelityGenerator
from xopt.generators.bayesian.upper_confidence_bound import (
    TDUpperConfidenceBoundGenerator,
    UpperConfidenceBoundGenerator,
)
import torch

# set default precision
torch.set_default_dtype(torch.double)


__all__ = [
    "BayesianExplorationGenerator",
    "MOBOGenerator",
    "UpperConfidenceBoundGenerator",
    "ExpectedImprovementGenerator",
    "MultiFidelityGenerator",
    "TDUpperConfidenceBoundGenerator",
    "TDExpectedImprovementGenerator",
]
