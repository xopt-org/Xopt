import torch

from xopt.generators.bayesian.bax_generator import BaxGenerator
from xopt.generators.bayesian.bayesian_exploration import BayesianExplorationGenerator
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.expected_improvement import (
    ExpectedImprovementGenerator,
    TDExpectedImprovementGenerator,
)
from xopt.generators.bayesian.mggpo import MGGPOGenerator
from xopt.generators.bayesian.mobo import MOBOGenerator
from xopt.generators.bayesian.multi_fidelity import MultiFidelityGenerator
from xopt.generators.bayesian.time_dependent import TimeDependentBayesianGenerator
from xopt.generators.bayesian.turbo import (
    EntropyTurboController,
    OptimizeTurboController,
    SafetyTurboController,
    TurboController,
)
from xopt.generators.bayesian.upper_confidence_bound import (
    TDUpperConfidenceBoundGenerator,
    UpperConfidenceBoundGenerator,
)

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
    "MGGPOGenerator",
    "BaxGenerator",
    "BayesianGenerator",
    "TimeDependentBayesianGenerator",
    "TurboController",
    "SafetyTurboController",
    "OptimizeTurboController",
    "EntropyTurboController",
]
