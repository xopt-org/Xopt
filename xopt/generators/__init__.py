from xopt.generators.bayesian import registry as bayes_registry
from xopt.generators.ga import registry as ga_registry
from xopt.generators.random import RandomGenerator
from xopt.generator import GeneratorOptions

registry = {
    "Random": (RandomGenerator, GeneratorOptions)
}
registry.update(**bayes_registry)
registry.update(**ga_registry)

