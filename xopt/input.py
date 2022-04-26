from typing import Tuple

from dependency_injector import providers, containers
from pydantic import BaseModel

import xopt.generators as gen
from xopt.evaluator import Evaluator, EvaluatorOptions
from xopt.generator import Generator, GeneratorOptions
from xopt.vocs import VOCS

import yaml


class Context(containers.DeclarativeContainer):
    config = providers.Configuration()

    mobo_generator = providers.Factory(
        gen.bayesian.MOBOGenerator,
        options=gen.bayesian.MOBOOptions(
            acq=gen.bayesian.mobo.MOBOAcqOptions(
                **(config.xopt.generator.pop("acq_options") or {})
            ),
            optim=gen.bayesian.options.OptimOptions(
                **(config.xopt.generator.pop("optim_options") or {})
            ),
            model=gen.bayesian.options.ModelOptions(
                **(config.xopt.generator.pop("model_options") or {})
            ),
        ),
    )

    evaluator = providers.Factory(
        Evaluator,
        options=EvaluatorOptions(**(config.xopt.evaluator.pop("options") or {})),
    )

    generator_selector = providers.Selector(
        config.xopt.generator.name,
        mobo=mobo_generator,
    )

    vocs = providers.Factory(VOCS, options=config.xopt.pop("vocs"))


def process_yaml(yaml_file) -> Tuple[Generator, Evaluator, VOCS]:
    context = Context()
    context.config.load_yaml(yaml_file)

    generator = context.generator_selector()
    evaluator = context.evaluator
    vocs = context.vocs

    return generator, evaluator, vocs


def assemble_defualt_yaml(generator_name: str) -> str:
    default_context = Context()
    default_context.config.xopt.generator = generator_name
    generator = default_context.generator_selector()
    generator_options = generator.options.to_json()

    return yaml.dump(generator_options)


class XoptOptions(BaseModel):
    # The version of xopt.
    version: str = "0.0.1"

    # xopt evaluator
    evaluator: EvaluatorOptions = EvaluatorOptions()

    # xopt vocs
    vocs: VOCS = VOCS()

    # xopt generator
    generator: GeneratorOptions = GeneratorOptions()

    # def __init__(self, **kwargs):
    #    super().__init__(**kwargs)
    #    self.check_for_duplicate_keys()

    def update(self, **kwargs):
        """
        Recursively update the options.
        """
        all_kwargs = kwargs

        def set_recursive(d: BaseModel):
            if not isinstance(d, dict):
                for name, val in d.__fields__.items():
                    attr = getattr(d, name)
                    if isinstance(attr, BaseModel):
                        set_recursive(attr)
                    elif name in kwargs.keys():
                        setattr(d, name, all_kwargs.pop(name))
                    else:
                        pass

        set_recursive(self)

        if len(all_kwargs):
            raise RuntimeError(
                f"keys {list(all_kwargs.keys())} not found, will not be " f"updated!"
            )

    def check_for_duplicate_keys(self):
        """
        Check if there are duplicate keys. If so, raise an error.
        """
        keys = self.get_all_keys()

        def check_duplicates(listOfElems):
            """Check if given list contains any duplicates"""
            setOfElems = set()
            for elem in listOfElems:
                if elem in setOfElems:
                    return True
                else:
                    setOfElems.add(elem)
            return False

        if check_duplicates(keys):
            raise RuntimeError(f"Duplicate keys found: invalid generator options!")

    def get_all_keys(self):
        """
        recursively get all keys of sub- and main-models.
        """
        keys = []
        for name, val in self.__fields__.items():
            attr = getattr(self, name)
            if isinstance(attr, XoptOptions):
                keys += attr.get_all_keys()
            else:
                keys += [name]

        return keys
