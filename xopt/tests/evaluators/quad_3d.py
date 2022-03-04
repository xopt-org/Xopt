import numpy as np
import torch
import time
from botorch.test_functions.multi_fidelity import AugmentedHartmann
import logging

logger = logging.getLogger(__name__)
VOCS = {
    'variables': {
        'x1': [0, 0.2],
        'x2': [0, 0.2],
        'x3': [0, 0.2],
        'cost': [0, 1.0]
    },
    'objectives': {
        'y1': 'MINIMIZE'

    }
}


# labeled version
def evaluate(inputs, extra_option='abc', **params):
    x = np.array((inputs['x1'], inputs['x2'], inputs['x3']))
    outputs = {'y1': np.linalg.norm(x - 0.15)**2}

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
