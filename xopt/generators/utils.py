# from xopt.generator import Generator
# from typing import Type
#
#
# def get_generator(name) -> Type[Generator]:
#     if name == "random":
#         from xopt.generators.random import RandomGenerator
#         return RandomGenerator
#     elif name == "upper_confidence_bound":
#         from xopt.generators.bayesian.upper_confidence_bound import \
#             UpperConfidenceBoundGenerator
#         return UpperConfidenceBoundGenerator
#     elif name == "mobo":
#         from xopt.generators.bayesian.mobo import MOBOGenerator
#         return MOBOGenerator
#     else:
#         raise ValueError(f"generator name {name} not found")
#
