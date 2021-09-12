import numpy as np
import torch
from botorch.test_functions.multi_fidelity import AugmentedHartmann

VOCS = {
    'name': '1D test',
    'description': '1D test function (with optional multi-fidelity) for debugging',
    'variables': {
        'x1': [0, 20.0],
        'cost': [0, 1.0]
    },
    'objectives': {
        'y1': 'MINIMIZE',

    },
    'constraints': {},
    'constants': {}
}


# labeled version
def evaluate(inputs, extra_option='abc', **params):
    x = inputs['x1']
    outputs = {'y1': (x - 15.0)**2}

    return outputs


if __name__ == '__main__':
    input = {'x1': 0.208,
             'x2': 0.164,
             'x3': 0.514,
             'x4': 0.280,
             'x5': 0.301,
             'x6': 0.664,
             'cost': 1.000}
    print(evaluate(input))
