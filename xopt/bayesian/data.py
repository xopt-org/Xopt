import json
import logging
import math
import numpy as np
import pandas as pd
from copy import deepcopy

# Logger
import torch

logger = logging.getLogger(__name__)


def save_data_dict(vocs, full_data, output_path):

    # add results to config dict and save to json
    results = {}

    names = ['variables', 'objectives', 'constraints']
    i = 0
    for name in names:
        val = {}
        for ele in vocs[name].keys():
            temp_data = full_data[:, i].tolist()
            # replace nans with None
            val[ele] = [x if not math.isnan(x) else None for x in temp_data]
            i += 1

        results[name] = val

    constraint_status = {}
    for ele in vocs['constraints'].keys():
        constraint_status[ele] = full_data[:, i].tolist()
        i += 1

    results['constraint_status'] = constraint_status
    results['feasibility'] = full_data[:, i].tolist()

    #outputs = deepcopy(config)
    #outputs['results'] = results
    output_path = '' if output_path is None else output_path
    with open(output_path + 'results.json', 'w') as outfile:
        json.dump(results, outfile)


def get_data_json(json_filename, vocs, **tkwargs):
    f = open(json_filename)
    data = json.load(f)

    data_sets = []
    names = ['variables', 'objectives', 'constraints']

    #replace None's with Nans
    def replace_none(l):
        return [math.nan if x is None else x for x in l]

    for name in names:
        data_sets += [torch.hstack([torch.tensor(replace_none(data['results'][name][ele]), **tkwargs).reshape(-1, 1) for
                                    ele in vocs[name].keys()])]

    return data_sets[0], data_sets[1], data_sets[2]


def save_data_pd(config, full_data):
    vocs = config['vocs']

    # add data to multi-index pandas array for storage
    names = ['variables', 'objectives', 'constraints']
    first_layer = []
    second_layer = []
    for name in names:
        for ele in vocs[name].keys():
            first_layer += [name]
            second_layer += [ele]

    for ele in vocs['constraints'].keys():
        first_layer += ['constraints']
        second_layer += [ele + '_stat']
    first_layer += ['constraints']
    second_layer += ['feasibility']

    index = pd.MultiIndex.from_tuples(list(zip(first_layer, second_layer)))

    df = pd.DataFrame(full_data.numpy(), columns=index)
    json_string = df.to_json()
    df_dict = json.loads(json_string)
    outputs = {'results': df_dict}
    outputs.update(config)

    if config['xopt']['output_path'] is None:
        output_path = ''
    else:
        output_path = config['xopt']['output_path']

    with open(output_path + 'results.json', 'w') as outfile:
        json.dump(outputs, outfile)
