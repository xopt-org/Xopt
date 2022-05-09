import importlib
import json
from copy import deepcopy
from typing import Dict, Tuple, Union

import pandas as pd
import yaml

from xopt.errors import XoptError
from xopt.evaluator import Evaluator
from xopt.generator import Generator

from xopt.options import XoptOptions
from xopt.utils import get_generator_and_defaults
from xopt.vocs import VOCS


def load_state_yaml(
    yaml_input: dict,
) -> Tuple[Generator, Evaluator, VOCS, XoptOptions, Union[pd.DataFrame, None]]:
    # get data and config from yaml dict
    data = None
    if "data" in yaml_input.keys():
        data = pd.read_json(json.dumps(yaml_input.pop("data")))

    generator, evaluator, vocs, options = read_config_dict(yaml_input)
    return generator, evaluator, vocs, options, data


def read_config_dict(config: dict) -> Tuple[Generator, Evaluator, VOCS, XoptOptions]:
    # read a yaml file and output objects for creating Xopt object

    # get copy of config
    config = deepcopy(config)

    options = XoptOptions(**config["xopt"])
    vocs = VOCS(**config["vocs"])

    # create generator
    generator_type, generator_options = get_generator_and_defaults(
        config["generator"].pop("name")
    )
    # TODO: use version number in some way
    if "version" in config["generator"].keys():
        config["generator"].pop("version")

    generator = generator_type(vocs, generator_options.parse_obj(config["generator"]))

    # create evaluator
    func = get_function(config["evaluator"]["function"])
    evaluator = Evaluator(func)

    return generator, evaluator, vocs, options


def state_to_dict(X):
    # dump data to dict with config metadata
    output = {
        "data": json.loads(X.data.to_json()),
        "xopt": json.loads(X.options.json()),
        "generator": {
            "name": X.generator.options.__config__.title,
            "version": X.generator.options.__config__.version,
            **json.loads(X.generator.options.json()),
        },
        "evaluator": json.loads(X.evaluator.options.json()),
        "vocs": json.loads(X.vocs.json()),
    }
    return output


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
