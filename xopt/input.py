import importlib
from typing import Tuple, Dict

import yaml

from xopt.options import XoptOptions
from xopt.evaluator import Evaluator
from xopt.generator import Generator
from xopt.utils import get_generator_and_defaults
from xopt.vocs import VOCS


def read_yaml(filename: str) -> Tuple[Generator, Evaluator, VOCS, XoptOptions]:
    with open(filename) as file:
        config = yaml.safe_load(file)

    return read_dict(config)


def read_dict(config) -> Tuple[Generator, Evaluator, VOCS, XoptOptions]:
    # read a yaml file and output objects for creating Xopt object

    options = XoptOptions(**config["xopt"])
    vocs = VOCS(**config["vocs"])

    # create generator
    generator_type, generator_options = get_generator_and_defaults(
        config["generator"]["name"]
    )
    generator = generator_type(
        vocs, generator_options.parse_obj(config["generator"]["options"])
    )

    # create evaluator
    func = get_function(config["evaluator"]["function"])
    evaluator = Evaluator(func)

    return generator, evaluator, vocs, options


def get_function(name):
    """
    Returns a function from a fully qualified name or global name.
    """

    # Check if already a function
    if callable(name):
        return name

    if not isinstance(name, str):
        raise ValueError(f"{name} must be callable or a string.")

    if name in globals():
        if callable(globals()[name]):
            f = globals()[name]
        else:
            raise ValueError(f"global {name} is not callable")
    else:
        if "." in name:
            # try to import
            m_name, f_name = name.rsplit(".", 1)
            module = importlib.import_module(m_name)
            f = getattr(module, f_name)
        else:
            raise Exception(f"function {name} does not exist")

    return f
