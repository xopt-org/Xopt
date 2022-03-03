"""
Tools to create datsets with pandas


"""

import pandas as pd
import json

KLIST = ['inputs', 'outputs', 'error']


def feasible(constraint_dict, output):
    """
    Use constraint dict and output dict to form a list of constraint evaluations.
    A constraint is satisfied if the evaluation is > 0.
    
    Returns a pandas Series of bools
    """
    results = {}
    for k in constraint_dict:
        x = output[k]
        op, d = constraint_dict[k]
        op = op.upper()  # Allow any case

        # Make a not null column
        results[k + '_notnull'] = ~x.isnull()

        if op == 'GREATER_THAN':  # x > d -> x-d > 0
            results[k] = x - d > 0
        elif op == 'LESS_THAN':  # x < d -> d-x > 0
            results[k] = d - x > 0
        else:
            raise ValueError(f'Unknown constraint operator: {op}')

    return pd.DataFrame(results).all(axis=1)


def load_xopt_data(xopt_json, verbose=False):
    """
    Load one JSON file, returns dict of DataFrame 
    """
    if verbose:
        print(xopt_json)
    indat = json.load(open(xopt_json))
    data = {'vocs': indat['vocs']}
    for k in KLIST:
        data[k] = pd.DataFrame(indat[k])

    return data


def load_all_xopt_data(xopt_json_list, verbose=False, add_feasible=True):
    """
    Loads many JSON files, concatenates and returns dict of DataFrame .
    
    If add_feasible, a 'feasible' column will be added according to the optimizer constraints.
    
    """
    dats = [load_xopt_data(f, verbose=verbose) for f in xopt_json_list]

    alldat = {}
    for k in KLIST:
        alldat[k] = pd.concat([d[k] for d in dats], ignore_index=True)

    vocs = dats[0]['vocs']

    # Concatenate
    cdat = pd.concat([alldat['inputs'], alldat['outputs']], axis=1)
    nc = len(cdat)
    if add_feasible and 'feasible' not in cdat:
        cdat['feasible'] = feasible(vocs["constraints"], cdat)
        if verbose:
            print(cdat['feasible'].sum(), 'feasible out of', len(cdat))

    return cdat
