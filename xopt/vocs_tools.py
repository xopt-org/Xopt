import numpy as np
import json
from math import isnan
from typing import Dict, Optional, Callable


# --------------------------------
# VOCS utilities
def save_vocs(vocs_dict, filePath=None):
    """
    Write VOCS dictionary to a JSON file. 
    If no filePath is given, the name is chosen from the 'name' key + '.json'
    """

    if filePath:
        name = filePath
    else:
        name = vocs_dict['name'] + '.json'
    with open(name, 'w') as outfile:
        json.dump(vocs_dict, outfile, ensure_ascii=True, indent='  ')
    print(name, 'written')


def load_vocs(filePath):
    """
    Load VOCS from a JSON file
    Returns a dict
    """
    with open(filePath, 'r') as f:
        dat = json.load(f)
    return dat


def skeys(some_dict):
    """
    Returns sorted keys
    
    Useful for converting dicts to lists consistently
    """
    return sorted([*some_dict])


OBJECTIVE_WEIGHT = {'MINIMIZE': -1.0, 'MAXIMIZE': 1.0}


def weight_list(objective_dict):
    """
    Returns a list of weights for use in optimization.
    The objective dict should be of the form:
        {'obj1':'minimize', 'obj2':'maximize'}
        
    The list is sorted by the keys.
    """

    weights = []
    for k in skeys(objective_dict):
        operator = objective_dict[k].upper()
        weights.append(OBJECTIVE_WEIGHT[operator])
    return weights


def get_bounds(vocs: Dict) -> np.ndarray:
    # get initial bounds
    return np.vstack(
        [np.array(ele) for _, ele in vocs.variables.items()]).T


# -------------------------------------------------------
# Output evaulation

def evaluate_objectives(objective_dict, output):
    """
    Uses objective dict and output dict to return a list of objective values,
    ordered by sorting the objective keys.
    """
    return [output[k] for k in skeys(objective_dict)]


def evaluate_constraints(constraint_dict, output):
    """
    Use constraint dict and output dict to form a list of constraint evaluations.
    A constraint is satisfied if the evaluation is > 0.
    """
    results = []
    for k in skeys(constraint_dict):
        x = output[k]
        op, d = constraint_dict[k]
        op = op.upper()  # Allow any case

        # check for nan
        if isnan(x):
            results.append(-666.0)
        elif op == 'GREATER_THAN':  # x > d -> x-d > 0
            results.append(x - d)
        elif op == 'LESS_THAN':  # x < d -> d-x > 0
            results.append(d - x)
        else:
            print('Unknown constraint operator:', op)
            raise
    return results


def constraint_satisfaction(constraint_dict, output):
    """
    Returns a dictionary of constraint names, and a bool with their satisfaction. 
    """
    vals = evaluate_constraints(constraint_dict, output)
    keys = skeys(constraint_dict)
    d = {}
    for k, v in zip(keys, vals):
        if v > 0:
            satisfied = True
        else:
            satisfied = False
        d[k] = satisfied
    return d


def n_constraints_satistfied(constraint_dict, output):
    vals = np.array(evaluate_constraints(constraint_dict, output))
    return len(np.where(vals > 0)[0])


def evaluate_constraints_inverse(constraint_dict, eval):
    """
    inverse of evaluate_constraints, returns a dictionary
    """
    output = {}
    con_names = skeys(constraint_dict)
    for name, val in zip(con_names, eval):
        op, d = constraint_dict[name]
        if op == 'GREATER_THAN':
            output[name] = val + d
        elif op == 'LESS_THAN':
            output[name] = d - val
        else:
            print('ERROR: unknown constraint operator: ', op)
            raise
    return output


def var_mins(var_dict):
    return [var_dict[name][0] for name in skeys(var_dict)]


def var_maxs(var_dict):
    return [var_dict[name][1] for name in skeys(var_dict)]


# -------------------------------------------------------
# Forming inputs


def inputs_from_vec(vec, vocs=None):
    """
    Forms labeled inputs from vector using vocs. If no vocs is given, labels are created in the form:
        variable_{i}

    """
    if not vocs:
        vkeys = [f'variable_{i}' for i in range(len(vec))]
        return dict(zip(vkeys, vec))

    # labeled inputs -> labeled outputs evaluate_f
    vkeys = skeys(vocs.variables)
    inputs = dict(zip(vkeys, vec))

    # Constants    
    if vocs.constants:
        inputs.update(vocs.constants or {})

    # Handle linked variables
    if vocs.linked_variables:
        for k, v in vocs.linked_variables.items():
            inputs[k] = inputs[v]

    return inputs


def vec_from_inputs(inputs, labels=None):
    """
    Forms vector from labeled inputs. 
    
    """
    if not labels:
        labels = skeys(inputs)
    return [inputs[k] for k in skeys(labels)]


def random_inputs(vocs, n=None, include_constants=True, include_linked_variables=True):
    """
    Uniform sampling of the variables described in vocs.variables = min, max.
    Returns a dict of inputs. 
    If include_constants, the vocs.constants are added to the dict. 
    
    
    Optional:
        n (integer) to make arrays of inputs, of size n. 
    
    """
    inputs = {}
    for key, val in vocs.variables.items():
        a, b = val
        x = np.random.random(n)
        inputs[key] = x * a + (1 - x) * b

    # Constants    
    if include_constants:
        inputs.update(vocs.constants)

    # Handle linked variables
    if include_linked_variables and vocs.linked_variables:
        for k, v in vocs.linked_variables.items():
            inputs[k] = inputs[v]

    return inputs
