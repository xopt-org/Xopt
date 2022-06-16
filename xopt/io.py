import json
import os
from copy import deepcopy
from typing import Dict, Tuple, Union

import pandas as pd
import yaml

from xopt.errors import XoptError
from xopt.evaluator import Evaluator, EvaluatorOptions
from xopt.generator import Generator

from xopt.options import XoptOptions
from xopt.utils import get_function, get_generator_and_defaults
from xopt.vocs import VOCS


def parse_config(config) -> dict:
    """
    Parse a config, which can be:
        YAML file
        JSON file
        dict-like object

    Returns a dict of kwargs for Xopt constructor.
    """
    if isinstance(config, str):
        if os.path.exists(config):
            yaml_str = open(config)
        else:
            yaml_str = config
        d = yaml.safe_load(yaml_str)
    else:
        d = config

    return xopt_kwargs_from_dict(d)


def xopt_kwargs_from_dict(config: dict) -> dict:
    """
    Processes a config dictionary and returns the corresponding Xopt kwargs.
    """

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

    # Create evaluator
    ev = config["evaluator"]
    ev["function"] = get_function(ev["function"])
    ev_options = EvaluatorOptions.parse_obj(ev)
    evaluator = Evaluator(**ev_options.dict())

    if "data" in config.keys():
        data = config["data"]
    else:
        data = None

    # return generator, evaluator, vocs, options, data
    return {
        "generator": generator,
        "evaluator": evaluator,
        "vocs": vocs,
        "options": options,
        "data": data,
    }


def state_to_dict(X, include_data=True):
    # dump data to dict with config metadata
    output = {
        "xopt": json.loads(X.options.json()),
        "generator": {
            "name": X.generator.alias,
            **json.loads(X.generator.options.json()),
        },
        "evaluator": json.loads(X.evaluator.options.json()),
        "vocs": json.loads(X.vocs.json()),
    }
    if include_data:
        output["data"] = json.loads(X.data.to_json())

    return output
