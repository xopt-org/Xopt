import numpy as np
import torch
from botorch.test_functions.multi_objective import ZDT1

VOCS = {
    'name': 'ZDT1_test',
    'description': 'ZDT1 unconstrained multi-objective test function',
    'simulation': 'ZDT1_test',
    'variables': {
        'x1': [0, 1.0],
        'x2': [0, 1.0]
    },
    'objectives': {
        'y1': 'MINIMIZE',
        'y2': 'MINIMIZE'

    },
    'constraints': {},
    'constants': {'a': 'dummy_constant'},
    'linked_variables': {'x9': 'x1'}

}

NAME = 'ZDT1'
BOUND_LOW, BOUND_UP = [0.0, 0.0], [1.0, 1.0]

X_RANGE = [0, 1.4]
Y_RANGE = [0, 1.4]


# labeled version
def evaluate(inputs, extra_option='abc', **params):
    info = {'some': 'info', 'about': ['the', 'run']}
    x = [inputs['x1'], inputs['x2']]

    if x[0] > BOUND_UP[0]:
        raise ValueError(f'Input greater than {BOUND_UP[0]} ')

    problem = ZDT1(2)
    objectives = problem.evaluate_true(torch.tensor(x)).numpy()
    outputs = {'y1': objectives[0], 'y2': objectives[1],
               'some_array': np.array([1, 2, 3])}

    return outputs
