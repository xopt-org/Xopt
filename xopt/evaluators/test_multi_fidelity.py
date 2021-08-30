import numpy as np
import torch
from botorch.test_functions.multi_fidelity import AugmentedHartmann

VOCS = {
    'name': 'AugmentedHartmann',
    'description': 'Augmented hartmann multi-fideldity optimization problem, provided by botorch, '
                   'see https://botorch.org/tutorials/multi_fidelity_bo',
    'variables': {
        'x1': [0, 1.0],
        'x2': [0, 1.0],
        'x3': [0, 1.0],
        'x4': [0, 1.0],
        'x5': [0, 1.0],
        'x6': [0, 1.0],
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
    x = [inputs[f'x{ele}'] for ele in range(1, 7)]
    x += [inputs['cost']]
    problem = AugmentedHartmann(negate=False)
    objective = problem(torch.tensor(x).reshape(1, -1)).numpy()
    outputs = {'y1': objective[0]}

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
