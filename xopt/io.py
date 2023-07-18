import pathlib

import pandas as pd
import yaml

from xopt import Xopt


def load_xopt_from_file(filename: str):
    file_extension = pathlib.Path(filename).suffix

    if file_extension == ".yml" or file_extension == ".yaml":
        return load_xopt_from_yaml(filename)


def load_xopt_from_yaml(filename: str):
    with open(filename, "r") as file:
        config = yaml.safe_load(file)
    return load_xopt_from_dict(config)


def load_xopt_from_dict(config: dict) -> Xopt:
    """
    Processes a config dictionary and returns the corresponding Xopt kwargs.
    """
    # return generator, evaluator, vocs, options, data
    if "data" in config:
        config["data"] = pd.DataFrame(config["data"])
    else:
        config["data"] = pd.DataFrame({})

    return Xopt(**config)
