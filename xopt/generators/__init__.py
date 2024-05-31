import json
import warnings
from typing import List

from xopt.errors import XoptError

from xopt.generators.random import RandomGenerator

# by default only load random generator
registered_generators = [
    RandomGenerator,
]

generators = {gen.name: gen for gen in registered_generators}

# This list hardcodes generator names - it is not pretty but helps with import speed A LOT
# don't import this directly -- use
all_generator_names = {
    "mggpo": {"mggpo"},
    "scipy": {"neldermead", "latin_hypercube"},
    "bo": {
        "upper_confidence_bound",
        "mobo",
        "bayesian_exploration",
        "time_dependent_upper_confidence_bound",
        "expected_improvement",
        "multi_fidelity",
    },
    "ga": {"cnsga"},
    "es": {"extremum_seeking"},
    "rcds": {"rcds"},
}


def list_available_generators() -> List[str]:
    try_load_all_generators()
    return list(generators.keys())


def try_load_all_generators():
    for v in all_generator_names.values():
        for gn in v:
            get_generator_dynamic(gn)


def get_generator_dynamic(name: str):
    if name in generators:
        return generators[name]

    if name in all_generator_names["mggpo"]:
        try:
            from xopt.generators.bayesian.mggpo import MGGPOGenerator

            generators[name] = MGGPOGenerator
            return MGGPOGenerator
        except ModuleNotFoundError:
            warnings.warn(
                "WARNING: `deap` and `botorch` not found, MGGPOGenerator is not "
                "available"
            )
    elif name in all_generator_names["scipy"]:
        try:
            from xopt.generators.scipy.latin_hypercube import LatinHypercubeGenerator
            from xopt.generators.scipy.neldermead import NelderMeadGenerator

            registered_generators = [
                NelderMeadGenerator,
                LatinHypercubeGenerator,
            ]

            for gen in registered_generators:
                generators[gen.name] = gen
            return generators[name]
        except ModuleNotFoundError:
            warnings.warn(
                "WARNING: `scipy` not found, NelderMeadGenerator and LatinHypercubeGenerator are not available"
            )
    elif name in all_generator_names["bo"]:
        from xopt.generators.bayesian.bayesian_exploration import (
            BayesianExplorationGenerator,
        )
        from xopt.generators.bayesian.expected_improvement import (
            ExpectedImprovementGenerator,
        )
        from xopt.generators.bayesian.mobo import MOBOGenerator
        from xopt.generators.bayesian.multi_fidelity import MultiFidelityGenerator
        from xopt.generators.bayesian.upper_confidence_bound import (
            TDUpperConfidenceBoundGenerator,
            UpperConfidenceBoundGenerator,
        )

        registered_generators = [
            UpperConfidenceBoundGenerator,
            MOBOGenerator,
            BayesianExplorationGenerator,
            TDUpperConfidenceBoundGenerator,
            ExpectedImprovementGenerator,
            MultiFidelityGenerator,
        ]
        for gen in registered_generators:
            generators[gen.name] = gen
        return generators[name]

    elif name in all_generator_names["ga"]:
        try:
            from xopt.generators.ga import CNSGAGenerator

            generators[name] = CNSGAGenerator
            return CNSGAGenerator
        except ModuleNotFoundError:
            warnings.warn("WARNING: `deap` not found, CNSGAGenerator is not available")
    elif name in all_generator_names["es"]:
        from xopt.generators.es.extremumseeking import ExtremumSeekingGenerator

        generators[name] = ExtremumSeekingGenerator
        return ExtremumSeekingGenerator
    elif name in all_generator_names["rcds"]:
        from xopt.generators.rcds.rcds import RCDSGenerator

        generators[name] = RCDSGenerator
        return RCDSGenerator
    raise KeyError


def get_generator(name: str):
    try:
        return get_generator_dynamic(name)
    except KeyError:
        raise XoptError(f"No generator named {name}")


def get_generator_defaults(
    name: str,
) -> dict:
    defaults = {}
    generator_class = get_generator(name)
    for k in generator_class.model_fields:
        if k in [
            "vocs",
            "data",
            "supports_batch_generation",
            "supports_multi_objective",
        ]:
            continue

        v = generator_class.model_fields[k]

        if v.exclude:
            continue

        if v.is_required():
            defaults[k] = None
        else:
            if v.default is None:
                defaults[k] = None
            else:
                try:
                    # handles pydantic models as defaults
                    defaults[k] = json.loads(v.default.json())
                except AttributeError:
                    # handles everything else
                    defaults[k] = v.default

    return defaults


#
# def get_generator_help(name):
#     generator = get_generator(name)
#     help_string = f"Generator name: {generator.name}\n"
#     help_string = recursive_description(generator, help_string)
#
#     return help_string
#
#
# def recursive_description(cls, help_string, in_key="", indent_level=0):
#     help_string += f"{in_key} : {cls}\n"
#     for key, val in cls.__fields__.items():
#         try:
#             if issubclass(val.type_, BaseModel):
#                 help_string = recursive_description(val.type_, help_string, key,
#                                                     indent_level+1)
#             else:
#                 help_string += "\t"*indent_level + f"{key} ({val.type_}):" \
#                                                    f" {val.field_info.description}\n"
#         except TypeError:
#             help_string += "\t"*indent_level + f"{key} ({val.type_}):" \
#                                                f" {val.field_info.description}\n"
#
#     return help_string
