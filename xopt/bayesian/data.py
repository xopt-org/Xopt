import json
import logging
import math
import os.path
from typing import Dict, Tuple, List

import pandas as pd

# Logger
import torch
from torch import Tensor

from .utils import (
    collect_results,
    get_feasability_constraint_status,
    NoValidResultsError,
)
from ..tools import NpEncoder

logger = logging.getLogger(__name__)


def gather_and_save_training_data(
    futures: List,
    vocs: Dict,
    tkwargs: Dict,
    train_x: Tensor = None,
    train_y: Tensor = None,
    train_c: Tensor = None,
    inputs: Dict = None,
    outputs: Dict = None,
    output_path: str = None,
) -> Tuple[Tensor, Tensor, Tensor, Dict, Dict]:
    if tkwargs is None:
        tkwargs = {}

    try:
        new_x, new_y, new_c, new_inputs, new_outputs = collect_results(
            futures, vocs, **tkwargs
        )

        if train_x is None:
            train_x = new_x
            train_y = new_y
            train_c = new_c
            inputs = new_inputs
            outputs = new_outputs

        else:
            # add new observations to training data
            train_x = torch.vstack((train_x, new_x))
            train_y = torch.vstack((train_y, new_y))

            if train_c is not None:
                train_c = torch.vstack((train_c, new_c))
            else:
                train_c = None

            inputs += new_inputs
            outputs += new_outputs

        # get feasibility values
        feas, constraint_status = get_feasability_constraint_status(
            train_y, train_c, vocs
        )

        if train_c is not None:
            elements = (train_x, train_y, train_c, constraint_status, feas)
        else:
            elements = (train_x, train_y, constraint_status, feas)

        full_data = torch.hstack(elements)
        logger.debug("saving data")
        save_data_dict(vocs, full_data, inputs, outputs, output_path)
        logger.debug("done")

    except NoValidResultsError:
        logger.warning("No valid results found, skipping to next iteration")

    return train_x, train_y, train_c, inputs, outputs


def save_data_dict(vocs, full_data, inputs, outputs, output_path):
    # add results to config dict and save to json
    results = {}

    vocs = dict(vocs) # Convert to dict so below still works. TODO: rework

    names = ["variables", "objectives"]

    if vocs["constraints"] is not None:
        names += ["constraints"]
    i = 0

    for name in names:
        val = {}
        for ele in vocs[name].keys():
            temp_data = full_data[:, i].tolist()
            # replace nans with None
            val[ele] = [x if not math.isnan(x) else None for x in temp_data]
            i += 1

        results[name] = val

    if vocs["constraints"] is not None:
        constraint_status = {}
        for ele in vocs["constraints"].keys():
            constraint_status[ele] = full_data[:, i].tolist()
            i += 1

        results["constraint_status"] = constraint_status
        results["feasibility"] = full_data[:, i].tolist()

    results["inputs"] = inputs
    results["outputs"] = outputs
    # outputs = deepcopy(config)
    # outputs['results'] = results
    output_path = "" if output_path is None else output_path
    # TODO: Combine into one function for xopt
    with open(os.path.join(output_path, "results.json"), "w") as outfile:
        json.dump(results, outfile, cls=NpEncoder)


def get_data_json(json_filename, vocs, **tkwargs):
    f = open(json_filename)
    data = json.load(f)

    data_sets = {}

    names = ["variables", "objectives", "constraints"]

    # replace None's with Nans
    def replace_none(l):
        return [math.nan if x is None else x for x in l]

    # TODO: rework this and unify with other algorithms
    for name in names:
        d = getattr(vocs, name)
        if d:
            data_sets[name] = torch.hstack(
                [
                    torch.tensor(replace_none(data[name][ele]), **tkwargs).reshape(
                        -1, 1
                    )
                    for ele in d.keys()
                ]
            )
        else:
            data_sets[name] = None
    data_sets["inputs"] = data["inputs"]
    data_sets["outputs"] = data["outputs"]

    return data_sets


def save_data_pd(config, full_data):
    vocs = config["vocs"]

    # add data to multi-index pandas array for storage
    names = ["variables", "objectives", "constraints"]
    first_layer = []
    second_layer = []
    for name in names:
        for ele in vocs[name].keys():
            first_layer += [name]
            second_layer += [ele]

    for ele in vocs.constraints.keys():
        first_layer += ["constraints"]
        second_layer += [ele + "_stat"]
    first_layer += ["constraints"]
    second_layer += ["feasibility"]

    index = pd.MultiIndex.from_tuples(list(zip(first_layer, second_layer)))

    df = pd.DataFrame(full_data.numpy(), columns=index)
    json_string = df.to_json()
    df_dict = json.loads(json_string)
    outputs = {"results": df_dict}
    outputs.update(config)

    if config["xopt"]["output_path"] is None:
        output_path = ""
    else:
        output_path = config["xopt"]["output_path"]

    with open(output_path + "results.json", "w") as outfile:
        json.dump(outputs, outfile)
