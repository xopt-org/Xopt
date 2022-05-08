from xopt.generators.bayesian import registry as bayes_registry
from xopt.generators.ga import registry as ga_registry
from xopt.generators.random import RandomGenerator, RandomOptions

registry = {
    "random": (RandomGenerator, RandomOptions)
}
registry.update(**bayes_registry)
registry.update(**ga_registry)

