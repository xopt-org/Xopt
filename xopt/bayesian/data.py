import json
import logging
import pandas as pd
from copy import deepcopy

# Logger
import torch

logger = logging.getLogger(__name__)


def save_data_dict(config, full_data):
    vocs = config['vocs']

    # add results to config dict and save to json
    results = {}

    names = ['variables', 'objectives', 'constraints']
    i = 0
    for name in names:
        val = {}
        for ele in vocs[name].keys():
            val[ele] = full_data[:, i].tolist()
            i += 1

        results[name] = val

    constraint_status = {}
    for ele in vocs['constraints'].keys():
        constraint_status[ele] = full_data[:, i].tolist()
        i += 1

    results['constraint_status'] = constraint_status
    results['feasibility'] = full_data[:, i].tolist()

    outputs = deepcopy(config)
    outputs['results'] = results

    if config['xopt']['output_path'] is None:
        output_path = ''
    else:
        output_path = config['xopt']['output_path']

    with open(output_path + 'results.json', 'w') as outfile:
        json.dump(outputs, outfile)


def get_data_json(json_filename, vocs, **tkwargs):
    f = open(json_filename)
    data = json.load(f)

    train_x = torch.hstack([torch.tensor(data['results']['variables'][ele], **tkwargs).reshape(-1, 1) for
                            ele in vocs['variables'].keys()])

    train_y = torch.hstack([torch.tensor(data['results']['objectives'][ele], **tkwargs).reshape(-1, 1) for
                                 ele in vocs['objectives'].keys()])

    train_c = torch.hstack([torch.tensor(data['results']['constraints'][ele], **tkwargs).reshape(-1, 1) for
                                 ele in vocs['constraints'].keys()])
    return train_x, train_y, train_c


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
