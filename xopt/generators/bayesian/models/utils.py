import pandas as pd


def split_data(data, vocs):
    variable_data = vocs.variable_data(data, "")
    objective_data = vocs.objective_data(data, "")
    constraint_data = vocs.constraint_data(data, "")
    return variable_data, objective_data, constraint_data


def get_keyed_index(data: pd.DataFrame, key: str):
    return list(data.keys()).index(key)
