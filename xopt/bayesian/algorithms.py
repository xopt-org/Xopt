from . import generators
from .optimize import bayesian_optimize


def mobo(vocs, evaluate_f, output_path=None, **kwargs):
    generator_options = kwargs.pop('generator_options', {})
    ref = kwargs.pop('ref', None)
    assert ref is not None, 'reference point required for MOBO, use keyword argument "ref"'

    generator = generators.mobo.MOBOGenerator(ref, **generator_options)
    return bayesian_optimize(vocs,
                             evaluate_f,
                             generator,
                             output_path,
                             **kwargs)


def bayesian_exploration(vocs, evaluate_f, output_path=None, **kwargs):
    generator_options = kwargs.pop('generator_options', {})
    generator = generators.exploration.BayesianExplorationGenerator(**generator_options)
    return bayesian_optimize(vocs,
                             evaluate_f,
                             generator,
                             output_path,
                             **kwargs)
